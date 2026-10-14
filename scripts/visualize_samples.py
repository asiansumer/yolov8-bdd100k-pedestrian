"""随机抽 N 张图, 把 YOLO 标签画回原图上, 人工检查转换是否正确。

为什么要做: 坐标转换只要错一点(比如 x/y 搞反、忘了除以宽高), 训练也能跑, 但结果会很差,
而且很难从 loss 上看出来。画出来看一眼是最便宜的检查方式。

用法:
  python scripts/visualize_samples.py --dataset-dir datasets/bdd100k_pedestrian --split val --n 8
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def read_yolo_labels(txt_path: Path, img_w: int, img_h: int) -> list:
    """读 YOLO 标签, 还原成原图像素坐标 (x1, y1, x2, y2)。"""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        # 与转换脚本相反的运算: 归一化中心点/宽高 -> 像素角点
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def main():
    p = argparse.ArgumentParser(description="可视化 YOLO 标签")
    p.add_argument("--dataset-dir", required=True, type=Path, help="转换输出目录 (含 images/ labels/)")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=8, help="抽多少张")
    p.add_argument("--only-with-ped", action="store_true", help="只抽含行人的图")
    p.add_argument("--seed", type=int, default=0, help="随机种子, 固定后每次抽到同样的图")
    p.add_argument("--out-dir", default=Path("reports/samples"), type=Path)
    args = p.parse_args()

    img_dir = args.dataset_dir / "images" / args.split
    lbl_dir = args.dataset_dir / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))
    if args.only_with_ped:
        images = [im for im in images if (lbl_dir / f"{im.stem}.txt").stat().st_size > 0]
    if not images:
        raise SystemExit(f"在 {img_dir} 没找到图片, 先运行 convert_bdd_to_yolo.py")

    random.seed(args.seed)
    picks = random.sample(images, min(args.n, len(images)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for im_path in picks:
        im = Image.open(im_path).convert("RGB")
        boxes = read_yolo_labels(lbl_dir / f"{im_path.stem}.txt", *im.size)
        draw = ImageDraw.Draw(im)
        for box in boxes:
            draw.rectangle(box, outline=(255, 0, 0), width=2)
        draw.text((10, 10), f"{im_path.name}  pedestrians: {len(boxes)}", fill=(255, 255, 0))
        im.save(args.out_dir / im_path.name)
        print(f"{im_path.name}: {len(boxes)} 个框")
    print(f"已保存 {len(picks)} 张到 {args.out_dir}")


if __name__ == "__main__":
    main()
