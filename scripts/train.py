"""按配置名训练 YOLOv8 行人检测模型。

用法:
  python scripts/train.py --config yolov8s                       # 正式训练 (默认 30 轮)
  python scripts/train.py --config yolov8s --epochs 1 --fraction 0.1   # 冒烟测试: 1 轮, 10% 训练集

结果保存在 runs/<name>/, 其中:
  weights/best.pt, weights/last.pt  模型权重
  results.csv                       每一轮的 loss / mAP50 / mAP50-95, 用来画曲线、排查"欠训练"
"""
import argparse

from ultralytics import YOLO

from common import REPO_ROOT, load_yaml, resolve_data_yaml


def main():
    p = argparse.ArgumentParser(description="训练 YOLOv8 (BDD100K 行人)")
    p.add_argument("--config", required=True, help="configs/ 下的文件名, 如 yolov8s / yolov8m / yolov8l")
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "bdd100k_pedestrian.yaml"))
    p.add_argument("--epochs", type=int, help="覆盖配置里的 epochs")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="只用训练集的这一比例 (0~1), 快速试跑用; 取的是排序后的前 x%%, "
                        "不是随机抽样; 验证集始终是完整的")
    p.add_argument("--batch", type=int, help="覆盖配置里的 batch (内存不够时调小)")
    p.add_argument("--device", default="mps", help="mps (Apple 芯片) / cpu / 0 (NVIDIA GPU)")
    p.add_argument("--name", help="实验名, 默认如 v8s_e30 (或 v8s_e1_f0.1)")
    args = p.parse_args()

    cfg = load_yaml(REPO_ROOT / "configs" / f"{args.config}.yaml")
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch

    # 实验名里写上模型大小和轮数, 以后对比不同实验时一眼就能看出来
    short = args.config.replace("yolov8", "v8")
    name = args.name or f"{short}_e{cfg['epochs']}" + (f"_f{args.fraction}" if args.fraction < 1 else "")

    model = YOLO(cfg.pop("model"))  # 加载 COCO 预训练权重 (迁移学习, 比从零训练收敛快得多)
    model.train(
        data=str(resolve_data_yaml(args.data)),
        device=args.device,
        fraction=args.fraction,
        project=str(REPO_ROOT / "runs"),
        name=name,
        exist_ok=False,  # 同名实验不覆盖, 自动加后缀 (v8s_e302), 防止误删旧结果
        plots=True,      # 保存 PR 曲线、混淆矩阵、训练曲线图, 面试时可以直接展示
        **cfg,           # 其余超参数 (lr0, batch, optimizer...) 全部来自配置文件
    )


if __name__ == "__main__":
    main()
