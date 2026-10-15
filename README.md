# YOLOv8 Pedestrian Detection on BDD100K (s / m / l)

[中文版](README_zh.md)

In 2025 I published a conference paper comparing YOLOv8s/m/l for pedestrian detection on BDD100K. At the time I was also preparing my graduation thesis defence and other lab work, so the models were only trained for 10 epochs. In this repo I redid the whole study: 30 epochs for every model, and the results split by lighting, weather, crowd density and object size.

The results were better than I expected. All three models beat the numbers in the paper by a clear margin, and the strange v8m result from the paper did not show up again. After splitting the results, the biggest problem turned out to be small pedestrians far away. Night-time made less difference than I thought it would.

## Key findings

1. The paper underestimated all three models. mAP50 went up by 0.12, 0.07 and 0.02 (s, m, l). When I did a first smoke test on my laptop with 10% of the data and 1 epoch, mAP50 was already 0.487, close to the paper's 0.534 after 10 epochs, and I thought something in my code was wrong. Comparing the logs later, the new runs were already well ahead at epoch 10, so the gap is not only about training longer (see [What changed](#what-changed-compared-with-the-paper)).
2. In the paper, v8m had a lower mAP50-95 than v8s (0.249 vs 0.263). This time the order is normal: s < m < l.
3. If I had to pick one model for a car, I would pick v8m. s to m costs 16% speed for +0.037 mAP50; m to l costs another 9% speed and 70% more parameters for only +0.015.
4. Small pedestrians (under 32×32 px) are detected badly: AP50 is only 0.41 to 0.48, while large ones are above 0.93. Day vs night is only about 0.06.
5. When I grouped by weather, *clear* scored lower than *overcast*, which confused me at first. A cross-check of weather against time of day showed that 61% of *clear* images were taken at night (you can't see clouds at night), compared with 1.5% of *overcast* images. So *clear* scores lower mostly because it has more night images, not because of the weather.

## Results

All numbers are on the full BDD100K validation set (10,000 images, 13,262 pedestrians), single class, 640 px input.

### Paper vs this repo

| Model | Params | Paper mAP50 | Paper mAP50-95 | **This repo mAP50** | **This repo mAP50-95** | Recall |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 11.1 M | 0.534 | 0.263 | **0.654** | **0.337** | 0.570 |
| YOLOv8m | 25.9 M | 0.618 | 0.249 | **0.691** | **0.368** | 0.613 |
| YOLOv8l | 43.6 M | 0.683 | 0.365 | **0.706** | **0.382** | 0.615 |

### Accuracy vs speed

| Model | mAP50 | Small-object AP50 | Night mAP50 | FPS (RTX 4090, batch 1, end-to-end) |
|---|---:|---:|---:|---:|
| YOLOv8s | 0.654 | 0.410 | 0.603 | **101.6** |
| YOLOv8m | 0.691 | 0.456 | 0.644 | 85.1 |
| YOLOv8l | **0.706** | **0.476** | **0.664** | 77.5 |

FPS includes image loading, preprocessing, inference and NMS. The paper's FPS (66 / 40 / 26) was measured on an Apple M3 Max, so absolute speeds are not comparable across the two; only the relative ordering is.

### Training curves

![mAP curves](reports/curves.png)

All three models plateau before epoch 30 (v8m and v8l by about epoch 20), so 30 epochs is close to converged for this setup.

### Breakdown by condition

mAP50 per group. Full tables, including mAP50-95 and recall: [`reports/eval_*/group_metrics.md`](reports).

| Group | Value | Images | Pedestrians | v8s | v8m | v8l |
|---|---|---:|---:|---:|---:|---:|
| Time of day | daytime | 5,258 | 9,366 | 0.669 | 0.707 | 0.720 |
| | night | 3,929 | 2,843 | 0.603 | 0.644 | 0.664 |
| | dawn/dusk | 778 | 1,046 | 0.655 | 0.676 | 0.693 |
| Weather | clear | 5,346 | 4,935 | 0.634 | 0.676 | 0.695 |
| | overcast | 1,239 | 1,916 | 0.671 | 0.706 | 0.723 |
| | rainy | 738 | 822 | 0.616 | 0.647 | 0.661 |
| | snowy | 769 | 1,137 | 0.702 | 0.733 | 0.753 |
| Crowd (≥ 10 people) | crowded | 351 | 5,255 | 0.666 | 0.704 | 0.714 |
| | not crowded | 9,649 | 8,007 | 0.650 | 0.687 | 0.705 |
| Object size* | small | | 6,399 | 0.410 | 0.456 | 0.476 |
| | medium | | 6,108 | 0.828 | 0.854 | 0.864 |
| | large | | 755 | 0.932 | 0.946 | 0.952 |

\* Size-group AP50 is computed by [`scripts/size_ap.py`](scripts/size_ap.py) (greedy matching at IoU 0.5, all-point interpolation, COCO area thresholds 32² / 96²). It is meant for comparing models within this repo, not for comparison with other papers.

Groups with fewer than 20 images (e.g. *foggy*, 13 images) are skipped, and the *undefined* time-of-day group (7 pedestrians) is too small to interpret.

**Some observations**

- All three models lose about 0.06 mAP50 at night. A bigger model improves day and night by about the same amount, so the gap stays. The training set is also unbalanced: night images are 40% of the data but only 21% of the pedestrian boxes.
- Weather and time of day are mixed together (see finding 5). Rain is a bit different: it has fewer small objects than average but still scores lowest of all weather types, so reflections and blur probably do matter.
- Crowded scenes score a bit higher on mAP50 but lower on mAP50-95 (v8l: 0.368 vs 0.394). My reading is that the model finds people in crowds but the boxes are less accurate because people block each other.
- Small objects are 48% of all pedestrians. The input is resized from 1280×720 to 640 px, so a 30 px tall pedestrian becomes about 15 px. Going from s to l improves small-object AP50 by 16%, but it is still far behind large objects.

## What changed compared with the paper

| | Paper | This repo |
|---|---|---|
| Epochs | 10 | 30 |
| Mosaic augmentation | If Ultralytics defaults were used, `close_mosaic=10` with 10 epochs switches mosaic off for the entire run | On for epochs 1–20, off for 21–30 |
| Label conversion | Original code lost | Rewritten and unit-tested ([`scripts/bdd_core.py`](scripts/bdd_core.py)) |
| Hardware | Apple M3 Max (MPS) | RTX 4090 (CUDA) |
| Optimizer, learning rate, batch size, image size | SGD, 640 px; v8s lr0 0.001 batch 8, v8l lr0 0.0005 batch 4 | Same (v8m, not specified in the paper: lr0 0.0007, batch 6) |

Validation mAP50 at the same point in training:

| Model | Paper, end of 10 epochs | This repo, epoch 10 of 30 | This repo, epoch 30 |
|---|---:|---:|---:|
| v8s | 0.534 | 0.631 | 0.652 |
| v8m | 0.618 | 0.676 | 0.690 |
| v8l | 0.683 | 0.694 | 0.704 |

Because the new runs are already ahead at epoch 10, the improvement is not only from training longer. Plausible contributors are mosaic being active and the different learning-rate schedule (a 30-epoch linear decay is still at a higher rate at epoch 10). I did not run an ablation to separate these, so this remains a hypothesis.

The effect of switching mosaic off is visible in the logs: training box loss drops sharply between epochs 20 and 21 (v8l: 1.487 → 1.239) while validation mAP barely moves, because the training images suddenly become easier while the test set does not change.

## Reproduce

Training was done on a rented RTX 4090. The whole pipeline (dependencies, data download, label conversion, tests) is one script:

```bash
git clone https://github.com/asiansumer/yolov8-bdd100k-pedestrian.git
cd yolov8-bdd100k-pedestrian
export KAGGLE_API_TOKEN=...          # BDD100K mirror on Kaggle, see note below
bash scripts/setup_cloud.sh

python scripts/train.py --config yolov8s --device 0 --name v8s_final   # ~7 h on a 4090
python scripts/evaluate_by_group.py --weights runs/v8s_final/weights/best.pt --device 0
python scripts/benchmark_fps.py --weights runs/v8s_final/weights/best.pt --device 0 --n 500
```

Replace `yolov8s` with `yolov8m` / `yolov8l` for the other models. Local setup on macOS is described in [SETUP.md](SETUP.md).

**Data source.** The official download site (bdd-data.berkeley.edu) was unreachable while this project was built, so the data came from a Kaggle mirror (`solesensei/solesensei_bdd100k`, 2018 label format). That mirror splits the training images into sub-folders; `setup_cloud.sh` flattens them. BDD100K's licence does not allow redistribution, so no images are included in this repo.

**Trained weights** are attached to the [GitHub release](https://github.com/asiansumer/yolov8-bdd100k-pedestrian/releases).

## Repository layout

```
scripts/
  bdd_core.py            BDD box -> YOLO format, size buckets (core logic, unit-tested)
  convert_bdd_to_yolo.py Converts a split and writes per-image metadata (weather, time, counts)
  setup_cloud.sh         One-command environment + data preparation
  train.py               Trains a model from configs/<name>.yaml
  evaluate_by_group.py   Per-scene and per-size evaluation
  size_ap.py             AP50 by object size
  benchmark_fps.py       End-to-end inference speed
  plot_curves.py         mAP-per-epoch comparison
  dataset_stats.py, visualize_samples.py   Data sanity checks
configs/                 Hyperparameters per model (same as the paper)
tests/                   Unit tests for label conversion and AP computation
reports/                 All result tables and figures in this README
```

## Limitations

- Each model was trained once, with no repeated seeds, so differences around 0.01 don't mean much.
- I did not run an ablation, so I can't say how much mosaic, the learning-rate schedule and the number of epochs each contributed.
- Some groups are small: rain has 822 pedestrians and dawn/dusk 1,046, compared with 9,366 for daytime. There were too few fog images to evaluate.
- The hardware is different from the paper, so FPS can only be compared within this repo.
- The data copy and conversion code are also different from the paper, which could affect the results too.

## Next steps

- Higher input resolution (e.g. 1280 px) aimed at small objects
- Multi-object tracking on BDD100K video using these detectors (next project)

## References

- Yaxiaer Yashengjiang. *A Comparative Analysis of Pedestrian Detection Performance for YOLOv8 Models on the BDD100K Dataset.* Conference paper, 2025.
- F. Yu et al. *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.* CVPR 2020.
- G. Jocher et al. *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
