"""把 BDD100K 检测标注(JSON)转换成 YOLO 格式, 只保留行人这一类。

同时支持两种官方格式:
  - 2020 "det_20": det_train.json / det_val.json, 行人类别名是 "pedestrian"
  - 2018 老格式:  bdd100k_labels_images_{train,val}.json, 行人类别名是 "person",
    并且车道线/可行驶区域用 poly2d 表示, 没有 box2d, 需要跳过

输出:
  out_dir/images/<split>/xxx.jpg   默认是指向原图的软链接(省 5GB+ 磁盘), 可用 --copy 复制
  out_dir/labels/<split>/xxx.txt   每行 "0 cx cy w h"(归一化到 0~1)
  out_dir/meta/<split>_meta.csv    每张图的天气/时间/场景/行人数/大中小目标数, 供分组评估使用

用法示例:
  python scripts/convert_bdd_to_yolo.py --labels det_val.json --images-dir .../images/100k/val \
      --out-dir datasets/bdd100k_pedestrian --split val
"""
import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

from bdd_core import box_to_yolo, frame_to_yolo, load_frames, size_bucket  # 核心转换逻辑在 bdd_core.py

DEFAULT_W, DEFAULT_H = 1280, 720  # BDD100K 所有图片都是 1280x720, 读不到尺寸时用这个兜底
META_FIELDS = ["image", "weather", "timeofday", "scene", "width", "height",
               "num_ped", "num_small", "num_medium", "num_large"]


def image_size(path: Path):
    """读取真实图片尺寸; PIL 只读文件头, 很快。读不到就用默认 1280x720。"""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return DEFAULT_W, DEFAULT_H


def main(argv=None):
    p = argparse.ArgumentParser(description="BDD100K JSON -> YOLO (单类: pedestrian)")
    p.add_argument("--labels", required=True, type=Path, help="标注 JSON 路径")
    p.add_argument("--images-dir", required=True, type=Path, help="原始图片目录, 如 images/100k/val")
    p.add_argument("--out-dir", required=True, type=Path, help="输出数据集根目录")
    p.add_argument("--split", required=True, help="train 或 val")
    p.add_argument("--categories", nargs="+", default=["pedestrian", "person"],
                   help="哪些类别算行人 (默认: pedestrian person)")
    p.add_argument("--copy", action="store_true", help="复制图片而不是建软链接")
    p.add_argument("--limit", type=int, default=0, help="只转换前 N 张, 用于快速测试 (0=全部)")
    args = p.parse_args(argv)

    categories = set(args.categories)
    img_out = args.out_dir / "images" / args.split
    lbl_out = args.out_dir / "labels" / args.split
    meta_out = args.out_dir / "meta"
    for d in (img_out, lbl_out, meta_out):
        d.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.labels)
    n_done = n_missing = n_boxes = 0
    size_total = {"small": 0, "medium": 0, "large": 0}
    meta_path = meta_out / f"{args.split}_meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        for frame in frames:
            if args.limit and n_done >= args.limit:
                break
            src = args.images_dir / frame["name"]
            if not src.exists():  # 标注里有但图片不存在(比如只下了部分数据), 跳过并计数
                n_missing += 1
                continue
            w, h = image_size(src)
            lines, areas = frame_to_yolo(frame, categories, w, h)

            # 没有行人的图也写一个空 txt: YOLO 会把它当"背景图", 有助于减少误检
            (lbl_out / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            dst = img_out / src.name
            if not dst.exists() and not dst.is_symlink():
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())  # 用绝对路径, 防止相对链接失效

            buckets = [size_bucket(a) for a in areas]
            for b in buckets:
                size_total[b] += 1
            attrs = frame.get("attributes") or {}
            writer.writerow({
                "image": src.name, "weather": attrs.get("weather", "undefined"),
                "timeofday": attrs.get("timeofday", "undefined"),
                "scene": attrs.get("scene", "undefined"), "width": w, "height": h,
                "num_ped": len(lines), "num_small": buckets.count("small"),
                "num_medium": buckets.count("medium"), "num_large": buckets.count("large"),
            })
            n_done += 1
            n_boxes += len(lines)

    print(f"[{args.split}] 标注帧数: {len(frames)}  已转换图片: {n_done}  缺失图片: {n_missing}")
    print(f"[{args.split}] 行人框总数: {n_boxes}  小/中/大: "
          f"{size_total['small']}/{size_total['medium']}/{size_total['large']}")
    print(f"[{args.split}] 元数据 CSV: {meta_path}")


if __name__ == "__main__":
    main()
