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
import json
import shutil
from pathlib import Path

from PIL import Image

DEFAULT_W, DEFAULT_H = 1280, 720  # BDD100K 所有图片都是 1280x720, 读不到尺寸时用这个兜底
SMALL_AREA = 32 * 32  # COCO 的大小目标阈值: 面积 < 32² 为小目标
MEDIUM_AREA = 96 * 96  # 32² ~ 96² 为中目标, >= 96² 为大目标
META_FIELDS = ["image", "weather", "timeofday", "scene", "width", "height",
               "num_ped", "num_small", "num_medium", "num_large"]


def size_bucket(area: float) -> str:
    """按 COCO 规则把框面积(原图像素)分到 small / medium / large。"""
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """把左上/右下角坐标转成 YOLO 的归一化 (cx, cy, w, h)。

    先裁剪到图像范围内(标注偶尔会越界一两个像素), 裁完宽或高 <= 0 的退化框返回 None。
    """
    x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
    y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    # YOLO 要的是"中心点 + 宽高", 并且都除以图像尺寸, 这样与输入分辨率无关
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, w / img_w, h / img_h)


def frame_to_yolo(frame: dict, categories: set, img_w: int, img_h: int):
    """处理一帧: 返回 (YOLO 标签行列表, 每个框的面积列表)。"""
    lines, areas = [], []
    # det_20 里没有目标的帧, labels 可能缺失或为 None, 用 "or []" 统一处理
    for label in frame.get("labels") or []:
        if label.get("category") not in categories:
            continue
        box = label.get("box2d")
        if not box:  # 老格式里 poly2d 标注没有 box2d, 直接跳过
            continue
        yolo = box_to_yolo(box["x1"], box["y1"], box["x2"], box["y2"], img_w, img_h)
        if yolo is None:
            continue
        lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo))  # 只有一类, 类别号固定为 0
        # 面积在"原图像素"上算(不是归一化后), 这样才能和 COCO 的 32²/96² 阈值对应
        areas.append(yolo[2] * img_w * yolo[3] * img_h)
    return lines, areas


def load_frames(json_path: Path) -> list:
    """读取 BDD100K 标注 JSON, 两种格式最外层都是"帧"的列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
