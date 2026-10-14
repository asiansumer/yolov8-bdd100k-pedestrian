"""读取转换脚本生成的元数据 CSV, 统计数据分布并输出 markdown 表格 + 柱状图。

为什么要做: 分组评估前先看清每组有多少图/多少框。比如 BDD100K 里雾天(foggy)图片非常少,
那一组的 mAP 波动会很大, 写结论时要谨慎。

用法:
  python scripts/dataset_stats.py --meta datasets/bdd100k_pedestrian/meta/val_meta.csv --out-dir reports/stats_val
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗口, 直接存图; 在没有显示器的服务器上也能运行
import matplotlib.pyplot as plt


def read_meta(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(rows: list) -> dict:
    """返回几个 Counter: 按时间段/天气统计图片数和行人框数, 以及大中小框数。"""
    stats = {"timeofday_images": Counter(), "timeofday_peds": Counter(),
             "weather_images": Counter(), "weather_peds": Counter(), "box_size": Counter()}
    for r in rows:
        n = int(r["num_ped"])
        stats["timeofday_images"][r["timeofday"]] += 1
        stats["timeofday_peds"][r["timeofday"]] += n
        stats["weather_images"][r["weather"]] += 1
        stats["weather_peds"][r["weather"]] += n
        for size in ("small", "medium", "large"):
            stats["box_size"][size] += int(r[f"num_{size}"])
    return stats


def to_markdown(stats: dict, n_images: int, n_with_ped: int) -> str:
    lines = [f"图片总数: {n_images}, 含行人的图片: {n_with_ped}", ""]
    for key in ("timeofday", "weather"):
        lines += [f"| {key} | 图片数 | 行人框数 |", "|---|---:|---:|"]
        for k, v in stats[f"{key}_images"].most_common():
            lines.append(f"| {k} | {v} | {stats[f'{key}_peds'][k]} |")
        lines.append("")
    total = sum(stats["box_size"].values()) or 1  # 防止除以 0
    lines += ["| 框大小 (COCO 阈值) | 框数 | 占比 |", "|---|---:|---:|"]
    for size in ("small", "medium", "large"):
        c = stats["box_size"][size]
        lines.append(f"| {size} | {c} | {c / total:.1%} |")
    return "\n".join(lines) + "\n"


def plot(stats: dict, out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("timeofday_images", "Images by time of day"),
              ("weather_images", "Images by weather"),
              ("box_size", "Pedestrian boxes by size")]
    for ax, (key, title) in zip(axes, panels):
        if key == "box_size":
            items = [(s, stats[key][s]) for s in ("small", "medium", "large")]  # 固定顺序更直观
        else:
            items = stats[key].most_common()
        ax.bar([k for k, _ in items], [v for _, v in items], color="#4C72B0")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="数据集分布统计")
    p.add_argument("--meta", required=True, type=Path, help="xxx_meta.csv 路径")
    p.add_argument("--out-dir", default=Path("reports/stats"), type=Path)
    args = p.parse_args()

    rows = read_meta(args.meta)
    stats = compute_stats(rows)
    n_with_ped = sum(1 for r in rows if int(r["num_ped"]) > 0)
    md = to_markdown(stats, len(rows), n_with_ped)
    print(md)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stats.md").write_text(md, encoding="utf-8")
    plot(stats, args.out_dir / "stats.png")
    print(f"已保存: {args.out_dir / 'stats.md'} 和 {args.out_dir / 'stats.png'}")


if __name__ == "__main__":
    main()
