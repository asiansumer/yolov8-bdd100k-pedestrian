"""对训练好的模型做分组评估: 按时间段 / 天气 / 是否拥挤, 以及按目标大小。

做法:
  - 场景分组: 用元数据 CSV 把验证集图片分组, 每组写一个"图片路径列表 txt"和一个临时数据集 yaml,
    再调用 Ultralytics 自带的 model.val() —— 它支持用 txt 列表当验证集, 所以 mAP 计算方式
    和整体评估完全一致, 结果可以直接和论文里的整体 mAP 对比。
  - 大小分组: Ultralytics 不按大小拆 mAP, 我们用 size_ap.py 里的简单匹配器自己算 AP50,
    具体做法和局限见 size_ap.py 开头的说明。

用法:
  python scripts/evaluate_by_group.py --weights runs/v8s_e30/weights/best.pt
输出: reports/eval_<实验名>/group_metrics.csv 和 group_metrics.md
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

from common import REPO_ROOT
from size_ap import match_image, size_ap, SIZES
from visualize_samples import read_yolo_labels

FIELDS = ["group", "value", "num_images", "num_ped", "mAP50", "mAP50-95", "recall"]


def write_subset_yaml(tmp_dir: Path, tag: str, image_paths: list, dataset_dir: Path) -> Path:
    """为一个子集写 txt 列表 + 数据集 yaml, 返回 yaml 路径。"""
    txt = tmp_dir / f"{tag}.txt"
    txt.write_text("\n".join(str(p) for p in image_paths) + "\n")
    cfg = {"path": str(dataset_dir), "train": str(txt), "val": str(txt), "names": {0: "pedestrian"}}
    yml = tmp_dir / f"{tag}.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    return yml


def run_val(model, data_yaml: Path, args) -> dict:
    m = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
                  device=args.device, plots=False, verbose=False)
    return {"mAP50": round(m.box.map50, 4), "mAP50-95": round(m.box.map, 4), "recall": round(m.box.mr, 4)}


def eval_by_size(model, image_paths: list, label_dir: Path, args) -> list:
    """逐张预测 + 与真值匹配, 最后按大小汇总 AP50。"""
    all_records, total_gt = [], defaultdict(int)
    # conf 设得很低 (0.001), 和 val 一致: 算 AP 需要完整的 P-R 曲线, 不能提前把低分框砍掉
    results = model.predict(source=[str(p) for p in image_paths], stream=True, conf=0.001,
                            iou=0.7, imgsz=args.imgsz, device=args.device, verbose=False)
    # 注意: Ultralytics 会把传入的图片列表重新排序, 所以不能用 zip(image_paths, results) 对齐,
    # 而要用每个结果自带的 r.path 找到对应的标签文件
    for r in results:
        h, w = r.orig_shape
        gts = read_yolo_labels(label_dir / f"{Path(r.path).stem}.txt", w, h)
        preds = [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        records, n_gt = match_image(preds, gts)
        all_records += records
        for s in SIZES:
            total_gt[s] += n_gt[s]
    res = size_ap(all_records, total_gt)
    return [{"group": "size", "value": s, "num_images": len(image_paths), "num_ped": res[s]["num_gt"],
             "mAP50": None if res[s]["ap50"] is None else round(res[s]["ap50"], 4), "mAP50-95": "",
             "recall": None if res[s]["recall"] is None else round(res[s]["recall"], 4)} for s in SIZES]


def main():
    p = argparse.ArgumentParser(description="分组评估 (场景 + 目标大小)")
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--dataset-dir", default=REPO_ROOT / "datasets" / "bdd100k_pedestrian", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--crowd-threshold", type=int, default=10, help="行人数 >= 这个值算'拥挤'(本仓库的假设)")
    p.add_argument("--min-images", type=int, default=20, help="图片数少于这个值的组跳过, 避免结论不可靠")
    p.add_argument("--size-max-images", type=int, default=0, help="大小评估只用前 N 张 (0=全部)")
    p.add_argument("--skip-size", action="store_true")
    p.add_argument("--out-dir", type=Path, help="默认 reports/eval_<实验名>")
    args = p.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    img_dir, label_dir = dataset_dir / "images" / args.split, dataset_dir / "labels" / args.split
    with open(dataset_dir / "meta" / f"{args.split}_meta.csv", newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    for r in meta:  # 派生一个"是否拥挤"的字段, 对应论文里的 crowded 场景
        r["crowd"] = "crowded" if int(r["num_ped"]) >= args.crowd_threshold else "not_crowded"

    run_name = args.weights.parent.parent.name  # runs/v8s_e30/weights/best.pt -> v8s_e30
    out_dir = args.out_dir or REPO_ROOT / "reports" / f"eval_{run_name}"
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    # 注意: 列表里用 images/<split>/ 下的路径 (而不是软链接指向的原图),
    # 因为 Ultralytics 靠把路径里的 /images/ 换成 /labels/ 来找标签
    all_paths = [img_dir / r["image"] for r in meta]
    rows = [{"group": "all", "value": "all", "num_images": len(meta),
             "num_ped": sum(int(r["num_ped"]) for r in meta),
             **run_val(model, write_subset_yaml(tmp_dir, "all", all_paths, dataset_dir), args)}]

    for key in ("timeofday", "weather", "crowd"):
        groups = defaultdict(list)
        for r in meta:
            groups[r[key]].append(r)
        for value, items in sorted(groups.items()):
            if len(items) < args.min_images:
                print(f"跳过 {key}={value}: 只有 {len(items)} 张图")
                continue
            # 文件名里去掉空格和斜杠: BDD100K 的取值里有 "dawn/dusk"、"partly cloudy"
            tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{key}_{value}")
            yml = write_subset_yaml(tmp_dir, tag,
                                    [img_dir / r["image"] for r in items], dataset_dir)
            rows.append({"group": key, "value": value, "num_images": len(items),
                         "num_ped": sum(int(r["num_ped"]) for r in items), **run_val(model, yml, args)})

    if not args.skip_size:
        paths = all_paths[: args.size_max_images] if args.size_max_images else all_paths
        rows += eval_by_size(model, paths, label_dir, args)

    with open(out_dir / "group_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md = ["| " + " | ".join(FIELDS) + " |", "|" + "---|" * len(FIELDS)]
    md += ["| " + " | ".join("" if r[k] is None else str(r[k]) for k in FIELDS) + " |" for r in rows]
    md.append("\n注: size 组的 mAP50 由 size_ap.py 自行计算 (全点插值), 与 Ultralytics 的数值不完全可比。")
    (out_dir / "group_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n已保存到 {out_dir}")


if __name__ == "__main__":
    main()
