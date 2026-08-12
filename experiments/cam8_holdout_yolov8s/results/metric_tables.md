### Cam8 hold-out test results

| Experiment | Test images | Test labels | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms/img) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp A · original only | 103 | 806 | 0.6690 | 0.6165 | 0.6417 | 0.6558 | 0.3087 | 2.44 |
| Exp B · CCTV Cam9+24 | 103 | 806 | 0.7920 | 0.7707 | 0.7812 | 0.8373 | 0.4100 | 3.27 |
| Exp C · original + CCTV | 103 | 806 | 0.8243 | 0.7431 | 0.7816 | 0.8319 | 0.4460 | 2.51 |

### Own-validation results (for reference, not the headline)

| Experiment | Val images | Val labels | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Exp A · original only | 100 | 350 | 0.9548 | 0.8686 | 0.9440 | 0.7074 |
| Exp B · CCTV Cam9+24 | 37 | 289 | 0.7717 | 0.7251 | 0.7944 | 0.4120 |
| Exp C · original + CCTV | 137 | 639 | 0.8723 | 0.8337 | 0.8980 | 0.6005 |

### Validation -> Cam8 generalisation gap

| Experiment | val mAP50 | Cam8 mAP50 | Δ mAP50 | val Recall | Cam8 Recall | Δ Recall |
|---|---:|---:|---:|---:|---:|---:|
| Exp A · original only | 0.9440 | 0.6558 | -0.2883 | 0.8686 | 0.6165 | -0.2520 |
| Exp B · CCTV Cam9+24 | 0.7944 | 0.8373 | +0.0429 | 0.7251 | 0.7707 | +0.0455 |
| Exp C · original + CCTV | 0.8980 | 0.8319 | -0.0661 | 0.8337 | 0.7431 | -0.0905 |
