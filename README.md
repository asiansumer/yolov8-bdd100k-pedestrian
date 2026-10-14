# YOLOv8 行人检测 @ BDD100K

> **English summary.** Reproduction and extension of my 2025 conference paper *"A Comparative Analysis of Pedestrian Detection Performance for YOLOv8 Models on the BDD100K Dataset"*. This repo converts BDD100K detection labels to a single-class (pedestrian) YOLO dataset, trains YOLOv8s/m/l (COCO-pretrained, SGD, 640 px) with Ultralytics 8.3.139 on Apple Silicon (MPS), and evaluates overall mAP, per-scene mAP (time of day / weather / crowded), per-object-size AP50 and inference FPS. Compared with the paper (10 epochs), the plan here is ≥30 epochs, with per-epoch logs kept so the paper's v8m mAP50-95 anomaly can be investigated.

## 项目简介

本项目复现并扩展我 2025 年的会议论文, 在 BDD100K 自动驾驶数据集上比较 YOLOv8s / m / l 三个规模的模型做**行人检测**的效果:

- 数据: BDD100K 官方划分, 约 7 万张训练图 / 1 万张验证图 (1280×720), 只保留"行人"一类
- 模型: YOLOv8s / m / l, COCO 预训练权重, 标准结构, SGD, imgsz=640, Ultralytics 默认数据增强
- 新增: 训练轮数从论文的 10 轮增加到 ≥30 轮; 按场景 (白天/夜晚/天气/拥挤) 和按目标大小 (小/中/大) 分组评估; FPS 测试; 单元测试

## 目录结构

```
yolov8-bdd100k-pedestrian/
├── README.md                  本文件
├── SETUP.md                   macOS 新手环境配置指南 (conda / MPS / Git / GitHub)
├── environment.yml            conda 环境 (名为 bdd, Python 3.10)
├── requirements.txt           pip 依赖 (固定 ultralytics==8.3.139)
├── configs/                   每个模型一份超参数
│   ├── yolov8s.yaml
│   ├── yolov8m.yaml
│   └── yolov8l.yaml
├── data/
│   └── bdd100k_pedestrian.yaml   Ultralytics 数据集配置
├── scripts/
│   ├── convert_bdd_to_yolo.py    BDD100K JSON -> YOLO txt + 元数据 CSV
│   ├── dataset_stats.py          数据分布统计 (表格 + 柱状图)
│   ├── visualize_samples.py      把标签画回图片, 检查转换是否正确
│   ├── train.py                  按配置训练
│   ├── plot_curves.py            多个实验的 mAP 曲线对比
│   ├── evaluate_by_group.py      分场景 / 分大小评估
│   ├── size_ap.py                按大小算 AP50 的小工具 (被上一个脚本调用)
│   ├── benchmark_fps.py          推理速度测试
│   └── common.py                 公共函数 (路径处理)
├── tests/                     pytest 单元测试 + 手写的小 JSON 样例
├── datasets/                  (不提交) 原始数据和转换后的数据
├── runs/                      (不提交) 训练输出: 权重、results.csv、曲线图
└── reports/                   评估结果表格和图 (可以提交, 用于展示)
```

## 从零开始的完整步骤

> 所有命令都在**仓库根目录**下运行。第一次配置 Mac 环境请先看 [SETUP.md](SETUP.md)。

### 1. 创建环境

```bash
conda env create -f environment.yml
conda activate bdd
python -c "import torch; print(torch.__version__, 'MPS可用:', torch.backends.mps.is_available())"
pytest -q          # 应显示全部测试通过
```

### 2. 下载 BDD100K

数据集需要在官网注册后才能下载: <https://bdd-data.berkeley.edu/>

登录后下载 **"100K Images"** 和 **检测标注 (Detection labels, 2020 版 det_20; 或 2018 版的 labels 也可以)**, 解压到 `datasets/bdd100k/`。解压后大致是下面这样 (以你实际解压出的文件夹名为准, 后面命令里的路径跟着改即可):

```
datasets/bdd100k/
├── images/100k/
│   ├── train/   约 7 万张 .jpg
│   ├── val/     1 万张 .jpg
│   └── test/    (不用)
└── labels/
    ├── det_20/det_train.json, det_val.json                 (2020 格式, 类别名 pedestrian)
    └── bdd100k_labels_images_train.json / _val.json         (2018 格式, 类别名 person) —— 二选一即可
```

> BDD100K 的许可证不允许再分发数据, `.gitignore` 已经排除了 `datasets/` 和所有图片, **千万不要把数据提交到 GitHub**。

### 3. 转换为 YOLO 格式

先用 `--limit` 转 200 张试一下, 确认没问题再全量转换:

```bash
python scripts/convert_bdd_to_yolo.py --labels datasets/bdd100k/labels/det_20/det_val.json \
    --images-dir datasets/bdd100k/images/100k/val --out-dir datasets/bdd_try --split val --limit 200
```

全量转换 (默认用软链接, 不会复制 5GB+ 的图片):

```bash
python scripts/convert_bdd_to_yolo.py --labels datasets/bdd100k/labels/det_20/det_train.json \
    --images-dir datasets/bdd100k/images/100k/train --out-dir datasets/bdd100k_pedestrian --split train
python scripts/convert_bdd_to_yolo.py --labels datasets/bdd100k/labels/det_20/det_val.json \
    --images-dir datasets/bdd100k/images/100k/val --out-dir datasets/bdd100k_pedestrian --split val
```

如果用 2018 格式的标注, 把 `--labels` 换成 `bdd100k_labels_images_train.json` 即可 (默认 `--categories pedestrian person` 两种名字都认)。

### 4. 检查数据

```bash
python scripts/dataset_stats.py --meta datasets/bdd100k_pedestrian/meta/val_meta.csv --out-dir reports/stats_val
python scripts/visualize_samples.py --dataset-dir datasets/bdd100k_pedestrian --split val --n 8 --only-with-ped
```

打开 `reports/samples/` 里的图, 确认红框确实框在行人身上。

### 5. 训练

```bash
# (a) 冒烟测试: 1 轮, 只用 10% 训练集, 确认整个流程能跑通 (每轮结束仍会在完整的 1 万张验证集上评估)
python scripts/train.py --config yolov8s --epochs 1 --fraction 0.1

# (b) 正式训练, 默认 30 轮, 输出到 runs/v8s_e30/
python scripts/train.py --config yolov8s
python scripts/train.py --config yolov8m      # 建议晚上挂着跑
python scripts/train.py --config yolov8l      # 建议晚上挂着跑

# 内存不够 (报 MPS out of memory) 时把 batch 调小:
python scripts/train.py --config yolov8l --batch 2
```

**关于耗时 (请务必先看):** 在 M3 Max 上用全部约 7 万张图训练 v8l 30 轮, 粗略估计可能需要**一天甚至更久**; v8s 会快很多。我无法给出准确时间, 所以推荐顺序是: 先跑 (a) 冒烟测试, 看 Ultralytics 打印的每轮预计时间, 再决定 s → m → l 的安排。训练中断后可以从 `last.pt` 续训 (会沿用原来的全部参数):

```bash
python -c "from ultralytics import YOLO; YOLO('runs/v8s_e30/weights/last.pt').train(resume=True)"
```

### 6. 评估

```bash
# 训练曲线对比 (排查"欠训练"最直观的方法)
python scripts/plot_curves.py runs/v8s_e30 runs/v8m_e30 runs/v8l_e30 --out reports/curves.png

# 分场景 + 分大小评估, 结果在 reports/eval_v8s_e30/group_metrics.{csv,md}
python scripts/evaluate_by_group.py --weights runs/v8s_e30/weights/best.pt

# 推理速度
python scripts/benchmark_fps.py --weights runs/v8s_e30/weights/best.pt --n 200
```

## 结果

### 基线 (论文, 10 epochs)

| 模型 | mAP50 | mAP50-95 | FPS (论文硬件) | lr0 | batch |
|---|---:|---:|---:|---:|---:|
| YOLOv8s | 0.5344 | 0.2625 | ~66 | 0.001 | 8 |
| YOLOv8m | 0.6181 | 0.2493 | ~40 | 未注明 | 未注明 |
| YOLOv8l | 0.6828 | 0.3650 | ~26 | 0.0005 | 4 |

论文中 v8l 小目标 mAP50 约 0.38, 比 v8s 高约 52%。

### 本仓库 (待填写)

环境: Ultralytics 8.3.139, PyTorch `____`, Apple M3 Max (MPS)

| 模型 | epochs | mAP50 | mAP50-95 | 小目标 AP50* | 夜晚 mAP50 | FPS (M3 Max, batch=1) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 30 | | | | | |
| YOLOv8m | 30 | | | | | |
| YOLOv8l | 30 | | | | | |

\* 小目标 AP50 由 `scripts/size_ap.py` 计算, 方法与论文/COCO 不完全相同, 只用于本仓库内 s/m/l 之间比较。

**假设说明:** 论文没有写 v8m 的超参数, 本仓库取 s 与 l 之间的值: lr0=0.0007, batch=6 (见 `configs/yolov8m.yaml`)。

## 已知问题

1. **mAP50-95 异常:** 论文中 v8m 的 mAP50 高于 v8s, 但 mAP50-95 (0.2493) 反而低于 v8s (0.2625)。mAP50-95 对框的定位精度要求更高, 更大的模型通常需要更多轮数才能把框"收紧"。猜测原因是 **10 轮训练不足 (欠训练)**, 本仓库通过 ≥30 轮训练 + `plot_curves.py` 对比每轮曲线来验证这个猜测。另外 v8m 的超参数论文未记录, 也可能是原因之一。
2. **只训练了 10 轮:** 对 7 万张图的数据集来说 10 轮偏少, 曲线大概率还没收敛, 论文结论应理解为"10 轮预算下的比较"。
3. **mosaic 增强可能没有生效:** Ultralytics 默认 `close_mosaic=10`, 即"最后 10 轮关闭 mosaic"。如果论文训练时总轮数也是 10 且用的默认值, 那 mosaic 从第 0 轮就被关掉了, 与论文中"使用了 mosaic"的描述不符。30 轮训练时前 20 轮有 mosaic、后 10 轮没有。
4. **FPS 不可直接比较:** 论文 FPS 与本仓库在 M3 Max 上测的 FPS 硬件不同, 只能比较同一台机器上的相对快慢。
5. **分组样本量:** BDD100K 中雾天 (foggy) 等少数类别图片很少, 这些组的 mAP 波动很大; `evaluate_by_group.py` 默认跳过少于 20 张图的组。"拥挤"定义为单图行人数 ≥10, 这是本仓库自己的假设 (`--crowd-threshold` 可调)。

## 测试

```bash
pytest -q
```

`tests/test_convert.py` 用两个手写的小 JSON (2020 格式和 2018 格式各一个) 验证: 坐标转换公式、类别过滤、越界裁剪、退化框丢弃、`labels` 为空、`poly2d` 跳过、元数据 CSV。`tests/test_size_ap.py` 验证 IoU 和 AP 计算。
