"""BDD100K 标注转换的核心函数 (我自己实现, 由 tests/test_convert.py 验证)。

- size_bucket:   按 COCO 标准把框面积分成 small / medium / large
- box_to_yolo:   BDD 的 (x1, y1, x2, y2) 像素坐标 -> YOLO 的归一化 (cx, cy, w, h)
- frame_to_yolo: 处理一张图的所有标注, 只保留行人, 返回 YOLO 标签行和每个框的像素面积
"""
import json
from pathlib import Path

SMALL_AREA = 32 * 32   # 面积 < 32²：小目标（COCO 标准）
MEDIUM_AREA = 96 * 96  # 32² ≤ 面积 < 96²：中目标；≥ 96²：大目标


def size_bucket(area: float) -> str:
    """输入框的面积（原图像素），返回 "small"、"medium" 或 "large"。"""

    if area < SMALL_AREA:
        return "small"
    elif area < MEDIUM_AREA:
        return "medium"
    else:
        return "large"



def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """把 BDD 的框（左上角 x1,y1，右下角 x2,y2，单位像素）转成 YOLO 格式。

    返回 (cx, cy, w, h)，四个数都是 0~1 之间的比例：
        cx, cy = 框中心点坐标 ÷ 图片宽 / 高
        w,  h  = 框的宽高   ÷ 图片宽 / 高

    两个特殊情况：
    1. 标注偶尔会超出图片边界几个像素，先把坐标限制在 [0, img_w] 和 [0, img_h] 之内。
       提示：max(0, min(x, img_w)) 能把 x 夹在 0 和 img_w 之间。
    2. 限制之后，如果宽或高 <= 0（框完全在图外，或者本身是一条线），返回 None。
    """

    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))

    if x2-x1 <= 0 or y2-y1 <= 0:
        return None

    cx = ((x1+x2)/2) / img_w
    cy = ((y1+y2)/2) / img_h
    w = (x2-x1) / img_w
    h = (y2-y1) / img_h
    return (cx, cy, w, h)



def frame_to_yolo(frame: dict, categories: set, img_w: int, img_h: int):
    """处理一张图的全部标注，返回 (lines, areas) 两个列表。

    frame 长这样（一张图的标注）：
        {"name": "xxx.jpg",
         "attributes": {"weather": "clear", "timeofday": "night", ...},
         "labels": [
            {"category": "person", "box2d": {"x1": 100, "y1": 200, "x2": 140, "y2": 300}},
            {"category": "car",    "box2d": {...}},
            {"category": "lane",   "poly2d": [...]},     # 车道线没有 box2d
         ]}

    要求：
    - 只保留 category 在 categories 里的标注
    - 没有 box2d 的跳过；box_to_yolo 返回 None 的也跳过
    - lines 里每一项是一行文字："0 cx cy w h"，数字保留 6 位小数，例如
          "0 0.093750 0.347222 0.031250 0.138889"
      开头的 0 是类别编号，我们只有行人一类，所以永远是 0。
      提示：f"{v:.6f}" 可以把数字格式化成 6 位小数。
    - areas 里每一项是这个框在原图上的像素面积（宽像素 × 高像素），后面用来分大中小目标
    - 注意："labels" 这个键可能不存在，也可能是 None，这两种情况都要当成没有标注
    """


    lines = []
    areas = []

    for label in frame.get("labels") or []:
        if label.get("category") not in categories:
            continue

        box =label.get("box2d")
        if not box:
            continue

        result = box_to_yolo(box["x1"], box["y1"], box["x2"], box["y2"], img_w, img_h)

        if result is None:
            continue

        cx, cy, w, h = result
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        areas.append(w*img_w*h*img_h)


    return lines, areas


def load_frames(json_path: Path) -> list:
    """读取标注 JSON（这个已经写好，测试会用到）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
