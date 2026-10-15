# Pedestrian Detection on BDD100K: Revisiting a YOLOv8 Size Comparison

[中文版](README_zh.md)

In 2025 I published a conference paper comparing YOLOv8s/m/l for pedestrian detection on BDD100K. The original experiments were trained for only 10 epochs under a tight deadline. This repo rebuilds that study from scratch as a reproducible pipeline, trains each model properly, and breaks the results down by lighting, weather, crowd density and object size.

**Short version:** with full training, every model beats the paper's numbers by a wide margin, the paper's odd v8m result disappears, and the main weakness turns out to be small, distant pedestrians rather than night-time driving.

## Key findings

1. **The paper underestimated all three models.** mAP50 rose by 0.12 (v8s), 0.07 (v8m) and 0.02 (v8l). Even at epoch 10 of the new runs, v8s and v8m were already well above the paper's 10-epoch results, so training length alone does not explain the gap (see [What changed](#what-changed-compared-with-the-paper)).
2. **The paper's v8m anomaly is gone.** In the paper, v8m had a *lower* mAP50-95 than v8s (0.249 vs 0.263). Here the ordering is s < m < l on every metric.
3. **v8m is the best trade-off.** Going from s to m costs 16% throughput for +0.037 mAP50. Going from m to l costs another 9% and 70% more parameters for only +0.015.
4. **Small objects are the real bottleneck.** AP50 on small pedestrians (< 32×32 px) is 0.41–0.48, against 0.93–0.95 on large ones. The day/night gap is about 0.06.
5. **"Clear weather performs worse than overcast" is a confounding artefact.** 61% of images labelled *clear* are night-time images (cloud cover is not visible at night), while only 1.5% of *overcast* images are. Lighting, not weather, explains the difference.

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

**What the breakdown shows**

- **Night costs about 0.06 mAP50 for every model.** A bigger model raises night and day performance by about the same amount; it does not close the gap. The training set is also imbalanced: night images are 40% of the data but contain only 21% of the pedestrian boxes.
- **Weather groups are confounded with time of day.** Clear-weather images are 61% night, overcast and partly-cloudy images are under 2% night. Rain is the one condition that scores lowest even though it has fewer small objects than average, so it likely has a genuine effect (reflections, blur).
- **Crowds are detected but localised less precisely.** Crowded scenes score slightly higher on mAP50 but lower on mAP50-95 (v8l: 0.368 vs 0.394), consistent with occlusion making box boundaries ambiguous.
- **Small objects dominate the error budget.** They are 48% of all pedestrians. Inputs are downscaled from 1280×720 to 640 px, so a 30 px pedestrian becomes about 15 px. Larger models help (+16% from s to l), but the gap to large objects remains huge.

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

- **Single run per model.** No repeated seeds, so differences of about 0.01 mAP should not be over-interpreted.
- **No ablation.** The contribution of mosaic vs learning-rate schedule vs epochs is not isolated.
- **Small groups are noisy.** Rain (822 pedestrians) and dawn/dusk (1,046) are much smaller than daytime (9,366). Fog could not be evaluated.
- **Different hardware from the paper**, so FPS is only comparable within this repo.
- **Different data copy and conversion code** from the paper, which may also contribute to the difference.

## Next steps

- Higher input resolution (e.g. 1280 px) aimed at small objects
- Multi-object tracking on BDD100K video using these detectors (next project)

## References

- Yaxiaer Yashengjiang. *A Comparative Analysis of Pedestrian Detection Performance for YOLOv8 Models on the BDD100K Dataset.* Conference paper, 2025.
- F. Yu et al. *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.* CVPR 2020.
- G. Jocher et al. *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
