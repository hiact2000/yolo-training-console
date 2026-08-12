"""Build the Cam8 hold-out dataset splits for the YOLOv8s cross-camera experiment.

Design rules (fixed, do not relax):
  * Cam8 is a pure hold-out test set. No Cam8 image may enter train or val.
  * Train : Val = 8 : 2, grouped by source image so the Roboflow duplicate copies of
    one photo never straddle the train/val boundary.
  * Exp C splits are the union of the Exp A and Exp B splits, so the three
    experiments differ only in which pools they draw from.
  * Nothing under datasets/ is copied, moved or modified. We only emit image-list
    txt files, dataset yamls and manifests.

Outputs (all under <out-root>/):
  manifests/*.csv   per-experiment train/val/test manifests + all_split_summary.csv
  lists/*.txt       Ultralytics image-list files referenced by the dataset yamls
  datasets/*.yaml   one dataset yaml per experiment
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The image data lives in the primary checkout; a git worktree does not carry it
# because datasets/ is gitignored. Point --data-root there when running from one.
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

ROBOFLOW_SUFFIX = re.compile(r"_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)
SEED = 42
VAL_FRACTION = 0.2


@dataclass
class Item:
    """One image plus its YOLO label."""

    image: Path
    label: Path
    origin: str  # "original" | "cctv0715"
    camera: str  # "cam8" | "cam9" | "cam24" | "non_cctv"
    group_key: str  # images sharing this key are never split apart
    stratum: str  # keeps the 8:2 ratio balanced per source kind
    rel_image: str
    rel_label: str
    n_boxes: int = 0
    md5: str = ""
    frame_index: int | None = None
    leakage_flags: list[str] = field(default_factory=list)


def classify_camera(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("cam8__") or re.search(r"cam0?8(?![0-9])", lowered):
        return "cam8"
    if lowered.startswith("cam9__") or re.search(r"cam0?9(?![0-9])", lowered):
        return "cam9"
    if lowered.startswith("cam24__") or re.search(r"cam24(?![0-9])", lowered):
        return "cam24"
    return "non_cctv"


def original_stratum(name: str, camera: str) -> str:
    """Sub-type of the original dataset, so the 8:2 split stays balanced."""
    if camera != "non_cctv":
        return f"original_{camera}"
    if name.startswith("P_"):
        return "original_phone"
    if name.startswith("video_frame"):
        return "original_videoframe"
    return "original_misc"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_boxes(label: Path) -> int:
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def collect(dirs: list[Path], origin: str, data_root: Path) -> tuple[list[Item], list[str]]:
    """Read every image/label pair under the given split folders."""
    items: list[Item] = []
    skipped: list[str] = []
    for split_dir in dirs:
        image_dir, label_dir = split_dir / "images", split_dir / "labels"
        for image in sorted(image_dir.glob("*.jpg")):
            label = label_dir / f"{image.stem}.txt"
            rel_image = image.relative_to(data_root).as_posix()
            if not label.exists():
                skipped.append(f"{rel_image} (missing label)")
                continue
            camera = classify_camera(image.stem)
            if origin == "original":
                group_key = f"original::{ROBOFLOW_SUFFIX.sub('', image.stem)}"
                stratum = original_stratum(image.stem, camera)
                frame_index = None
            else:
                group_key = f"cctv::{image.stem}"
                stratum = f"cctv_{camera}"
                match = re.search(r"_(\d{5})$", image.stem)
                frame_index = int(match.group(1)) if match else None
            items.append(
                Item(
                    image=image,
                    label=label,
                    origin=origin,
                    camera=camera,
                    group_key=group_key,
                    stratum=stratum,
                    rel_image=rel_image,
                    rel_label=label.relative_to(data_root).as_posix(),
                    n_boxes=count_boxes(label),
                    md5=md5_of(image),
                    frame_index=frame_index,
                )
            )
    return items, skipped


def grouped_split(items: list[Item], seed: int) -> tuple[list[Item], list[Item]]:
    """Split 8:2 by group_key, keeping each stratum's ratio intact.

    Every image sharing a group_key lands on the same side, which is what stops the
    Roboflow triplicates from leaking between train and val.
    """
    by_stratum: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_stratum[item.stratum][item.group_key].append(item)

    train: list[Item] = []
    val: list[Item] = []
    for stratum in sorted(by_stratum):
        groups = sorted(by_stratum[stratum])
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * VAL_FRACTION)) if len(groups) > 1 else 0
        for position, key in enumerate(groups):
            (val if position < n_val else train).extend(by_stratum[stratum][key])
    return train, val


def flag_leakage(train: list[Item], val: list[Item], test: list[Item]) -> None:
    """Annotate images that could make the split optimistic. Annotate only, never drop."""
    named = (("train", train), ("val", val), ("test", test))
    split_of: dict[int, str] = {}
    for split_name, bucket in named:
        for item in bucket:
            split_of[id(item)] = split_name

    # 1. Identical file content appearing in more than one split.
    by_md5: dict[str, set[str]] = defaultdict(set)
    for split_name, bucket in named:
        for item in bucket:
            by_md5[item.md5].add(split_name)
    for _, bucket in named:
        for item in bucket:
            if len(by_md5[item.md5]) > 1:
                item.leakage_flags.append("duplicate_content_across_splits")

    # 2. Roboflow multi-copy groups: mitigated, since grouped_split keeps them together.
    group_sizes: Counter[str] = Counter()
    for _, bucket in named:
        for item in bucket:
            group_sizes[item.group_key] += 1
    for _, bucket in named:
        for item in bucket:
            if item.origin == "original" and group_sizes[item.group_key] > 1:
                item.leakage_flags.append("roboflow_multicopy_grouped_ok")

    # 3. CCTV frames whose immediate neighbour in the same video sits in the other split.
    by_video: dict[str, dict[int, str]] = defaultdict(dict)
    for split_name, bucket in named:
        for item in bucket:
            if item.origin == "cctv0715" and item.frame_index is not None:
                video = item.image.stem.rsplit("_", 1)[0]
                by_video[video][item.frame_index] = split_name
    for _, bucket in named:
        for item in bucket:
            if item.origin != "cctv0715" or item.frame_index is None:
                continue
            video = item.image.stem.rsplit("_", 1)[0]
            here = split_of[id(item)]
            neighbours = by_video[video]
            if any(neighbours.get(item.frame_index + off, here) != here for off in (-1, 1)):
                item.leakage_flags.append("adjacent_video_frame_other_split")


def write_manifest(path: Path, items: list[Item], experiment: str, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "split",
                "image_path",
                "label_path",
                "origin",
                "camera",
                "stratum",
                "group_key",
                "n_boxes",
                "md5",
                "leakage_risk",
            ]
        )
        for item in sorted(items, key=lambda i: i.rel_image):
            writer.writerow(
                [
                    experiment,
                    split,
                    item.rel_image,
                    item.rel_label,
                    item.origin,
                    item.camera,
                    item.stratum,
                    item.group_key,
                    item.n_boxes,
                    item.md5,
                    "|".join(sorted(set(item.leakage_flags))),
                ]
            )


def write_list(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(str(i.image) for i in items)) + "\n", encoding="utf-8")


def write_yaml(path: Path, train_list: Path, val_list: Path, test_list: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# {note}",
                "# Auto-generated by scripts/build_cam8_holdout.py -- do not hand-edit.",
                "# Cam8 appears only in `test`; it is absent from train and val by construction.",
                "#",
                "# `path` below is machine-specific and the lists/ it points at are gitignored,",
                "# so a fresh checkout must regenerate both before training:",
                "#   python scripts/build_cam8_holdout.py --seed 42 --data-root <checkout with datasets/>",
                f"path: {path.parent.parent.as_posix()}",
                f"train: {train_list.relative_to(path.parent.parent).as_posix()}",
                f"val: {val_list.relative_to(path.parent.parent).as_posix()}",
                f"test: {test_list.relative_to(path.parent.parent).as_posix()}",
                "",
                "nc: 1",
                "names: ['comb']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def summarise(experiment: str, split: str, items: list[Item]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_camera: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_camera[item.camera].append(item)
    for camera in sorted(by_camera) + ["ALL"]:
        bucket = items if camera == "ALL" else by_camera[camera]
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "camera": camera,
                "n_images": len(bucket),
                "n_boxes": sum(i.n_boxes for i in bucket),
                "n_empty_labels": sum(1 for i in bucket if i.n_boxes == 0),
                "n_unique_groups": len({i.group_key for i in bucket}),
                "n_leakage_flagged": sum(1 for i in bucket if i.leakage_flags),
            }
        )
    return rows


EXPERIMENT_FILES = {
    "expA_original": (
        "expA_original_train.csv",
        "expA_original_val.csv",
        "expA_cam8_test.csv",
        "expA_original_cam8test.yaml",
    ),
    "expB_cctv_cam9_24": (
        "expB_cctv_cam9_24_train.csv",
        "expB_cctv_cam9_24_val.csv",
        "expB_cam8_test.csv",
        "expB_cctv_cam9_24_cam8test.yaml",
    ),
    "expC_original_cctv_cam9_24": (
        "expC_original_cctv_cam9_24_train.csv",
        "expC_original_cctv_cam9_24_val.csv",
        "expC_cam8_test.csv",
        "expC_original_cctv_cam9_24_cam8test.yaml",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Checkout that actually holds datasets/detect (default: primary checkout).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "cam8_holdout_yolov8s",
        help="Where manifests, lists and dataset yamls are written.",
    )
    args = parser.parse_args()

    detect_root = args.data_root / "datasets" / "detect"
    if not detect_root.is_dir():
        raise SystemExit(f"detect dataset not found: {detect_root}")

    original, skipped_original = collect(
        [detect_root / d for d in ("train", "valid", "test")], "original", args.data_root
    )
    cctv, skipped_cctv = collect(
        [detect_root / "0715data" / d for d in ("train", "valid")], "cctv0715", args.data_root
    )

    # Cam8 hold-out: every Cam8 frame from both sources, and nothing else.
    cam8_test = [i for i in original + cctv if i.camera == "cam8"]
    # Training pools, with Cam8 removed at the source.
    original_pool = [i for i in original if i.camera != "cam8"]
    cctv_pool = [i for i in cctv if i.camera in {"cam9", "cam24"}]

    a_train, a_val = grouped_split(original_pool, args.seed)
    b_train, b_val = grouped_split(cctv_pool, args.seed)
    c_train, c_val = a_train + b_train, a_val + b_val

    flag_leakage(c_train, c_val, cam8_test)

    experiments = {
        "expA_original": (a_train, a_val, "Experiment A -- original dataset only, Cam8 hold-out test"),
        "expB_cctv_cam9_24": (
            b_train,
            b_val,
            "Experiment B -- CCTV Cam9 + Cam24 only, Cam8 hold-out test",
        ),
        "expC_original_cctv_cam9_24": (
            c_train,
            c_val,
            "Experiment C -- original + CCTV Cam9/Cam24, Cam8 hold-out test",
        ),
    }

    summary_rows: list[dict[str, object]] = []
    for key, (train, val, note) in experiments.items():
        train_csv, val_csv, test_csv, yaml_name = EXPERIMENT_FILES[key]
        write_manifest(args.out_root / "manifests" / train_csv, train, key, "train")
        write_manifest(args.out_root / "manifests" / val_csv, val, key, "val")
        write_manifest(args.out_root / "manifests" / test_csv, cam8_test, key, "test")

        train_list = args.out_root / "lists" / f"{key}_train.txt"
        val_list = args.out_root / "lists" / f"{key}_val.txt"
        test_list = args.out_root / "lists" / "cam8_test.txt"
        write_list(train_list, train)
        write_list(val_list, val)
        write_list(test_list, cam8_test)
        write_yaml(args.out_root / "datasets" / yaml_name, train_list, val_list, test_list, note)

        for split_name, bucket in (("train", train), ("val", val), ("test", cam8_test)):
            summary_rows.extend(summarise(key, split_name, bucket))

    summary_path = args.out_root / "manifests" / "all_split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---- hard assertions: the hold-out rule must hold, loudly ----
    test_md5 = {i.md5 for i in cam8_test}
    for key, (train, val, _) in experiments.items():
        offenders = [i.rel_image for i in train + val if i.camera == "cam8"]
        assert not offenders, f"{key}: Cam8 leaked into train/val: {offenders[:5]}"
        overlap = sorted({i.rel_image for i in train + val if i.md5 in test_md5})
        assert not overlap, f"{key}: train/val shares file bytes with Cam8 test: {overlap[:5]}"
        shared_groups = {i.group_key for i in train} & {i.group_key for i in val}
        assert not shared_groups, f"{key}: group straddles train/val: {sorted(shared_groups)[:5]}"

    print(f"seed={args.seed}  val_fraction={VAL_FRACTION}  data_root={args.data_root}")
    print(f"skipped (no label): {len(skipped_original) + len(skipped_cctv)}")
    for entry in skipped_original + skipped_cctv:
        print(f"  - {entry}")
    print(
        f"Cam8 hold-out test: {len(cam8_test)} images, "
        f"{sum(i.n_boxes for i in cam8_test)} boxes, "
        f"{len({i.camera for i in cam8_test})} camera(s)"
    )
    for key, (train, val, _) in experiments.items():
        print(
            f"{key:28s} train={len(train):4d} val={len(val):4d} "
            f"groups={len({i.group_key for i in train})}/{len({i.group_key for i in val})} "
            f"boxes={sum(i.n_boxes for i in train)}/{sum(i.n_boxes for i in val)}"
        )
    print(f"leakage-flagged rows: {sum(1 for i in c_train + c_val + cam8_test if i.leakage_flags)}")
    print(f"outputs under: {args.out_root}")


if __name__ == "__main__":
    main()
