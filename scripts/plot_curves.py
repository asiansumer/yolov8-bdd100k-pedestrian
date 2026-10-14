"""把多个实验的 results.csv 画在同一张图上, 对比 mAP50 / mAP50-95 随轮数的变化。

为什么要做: 论文里 v8m 的 mAP50-95 反而低于 v8s, 怀疑是 10 轮"欠训练"。
把每轮的曲线画出来, 就能看到: 曲线在第 10 轮是否还在明显上升? 30 轮后 m 是否超过 s?

用法:
  python scripts/plot_curves.py runs/v8s_e30 runs/v8m_e30 runs/v8l_e30 --out reports/curves.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]  # Ultralytics results.csv 里的列名


def read_results(run_dir: Path) -> dict:
    """读取 results.csv, 返回 {列名: [每轮的数值]}。"""
    with open(run_dir / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    # 有些版本的列名带前导空格, 统一 strip 掉
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    return {k: [float(r[k]) for r in rows] for k in ["epoch"] + METRICS}


def main():
    p = argparse.ArgumentParser(description="对比多个实验的训练曲线")
    p.add_argument("runs", nargs="+", type=Path, help="实验目录, 如 runs/v8s_e30")
    p.add_argument("--out", default=Path("reports/curves.png"), type=Path)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in args.runs:
        res = read_results(run)
        for ax, m in zip(axes, METRICS):
            ax.plot(res["epoch"], res[m], marker=".", label=run.name)
        print(f"{run.name}: 最后一轮 mAP50={res[METRICS[0]][-1]:.4f}  mAP50-95={res[METRICS[1]][-1]:.4f}")
    for ax, m in zip(axes, METRICS):
        ax.axvline(10, color="gray", linestyle="--", linewidth=1)  # 论文只训到第 10 轮, 画条参考线
        ax.set_xlabel("epoch")
        ax.set_title(m)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
