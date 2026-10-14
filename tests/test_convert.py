"""转换脚本的单元测试。运行: pytest -q

用两个手写的小 JSON (tests/fixtures/) 分别模拟 2020 det_20 格式和 2018 老格式,
检查: 坐标换算、类别过滤、越界裁剪、退化框丢弃、labels 为 None、poly2d 跳过、元数据 CSV。
"""
import csv
import importlib
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# 默认测试正式脚本; 设置环境变量 CONV_MODULE=bdd_core 可以只测试核心函数
conv = importlib.import_module(os.environ.get("CONV_MODULE", "convert_bdd_to_yolo"))

FIX = Path(__file__).parent / "fixtures"
CATS = {"pedestrian", "person"}


def test_box_to_yolo_math():
    # 框 (100,200)-(140,300) 在 1280x720 图上: 中心 (120,250), 宽高 (40,100)
    cx, cy, w, h = conv.box_to_yolo(100, 200, 140, 300, 1280, 720)
    assert cx == pytest.approx(120 / 1280)
    assert cy == pytest.approx(250 / 720)
    assert w == pytest.approx(40 / 1280)
    assert h == pytest.approx(100 / 720)


def test_box_clipped_and_degenerate():
    # 越界的框被裁剪到图像边缘: x 1250~1280, y 600~720
    cx, cy, w, h = conv.box_to_yolo(1250, 600, 1300, 760, 1280, 720)
    assert (cx, cy) == pytest.approx((1265 / 1280, 660 / 720))
    assert (w, h) == pytest.approx((30 / 1280, 120 / 720))
    assert conv.box_to_yolo(50, 50, 50, 90, 1280, 720) is None  # 宽为 0
    assert conv.box_to_yolo(1300, 10, 1400, 50, 1280, 720) is None  # 完全在图外


def test_size_bucket_coco_thresholds():
    assert conv.size_bucket(31 * 31) == "small"
    assert conv.size_bucket(32 * 32) == "medium"
    assert conv.size_bucket(96 * 96) == "large"


def test_det20_format_filtering():
    frames = conv.load_frames(FIX / "det20_sample.json")
    lines, areas = conv.frame_to_yolo(frames[0], CATS, 1280, 720)
    # 5 个标注: 2 个有效行人; car 和 rider 被过滤; 宽为 0 的行人被丢弃
    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)
    assert lines[0] == "0 0.093750 0.347222 0.031250 0.138889"
    assert areas == pytest.approx([40 * 100, 30 * 120])
    # labels 为 None 的帧不报错, 返回空
    assert conv.frame_to_yolo(frames[1], CATS, 1280, 720) == ([], [])


def test_2018_format_person_and_poly2d():
    frames = conv.load_frames(FIX / "labels2018_sample.json")
    lines, _ = conv.frame_to_yolo(frames[0], CATS, 1280, 720)
    assert len(lines) == 1  # lane 和"没有 box2d 的 person"都被跳过
    # 只把 pedestrian 当行人时, 老格式的 person 一个都不保留
    assert conv.frame_to_yolo(frames[0], {"pedestrian"}, 1280, 720) == ([], [])


@pytest.mark.skipif(not hasattr(conv, "main"), reason="练习版只实现了核心函数, 没有 main")
def test_main_end_to_end(tmp_path):
    img_dir = tmp_path / "raw"
    img_dir.mkdir()
    Image.new("RGB", (1280, 720)).save(img_dir / "frame_a.jpg")  # 故意不创建 frame_b, 测试"缺失图片"
    out = tmp_path / "out"
    conv.main(["--labels", str(FIX / "det20_sample.json"), "--images-dir", str(img_dir),
               "--out-dir", str(out), "--split", "val"])

    assert (out / "images" / "val" / "frame_a.jpg").is_symlink()
    assert len((out / "labels" / "val" / "frame_a.txt").read_text().splitlines()) == 2
    assert not (out / "labels" / "val" / "frame_b.txt").exists()
    with open(out / "meta" / "val_meta.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["timeofday"] == "daytime"
    assert (rows[0]["num_ped"], rows[0]["num_small"], rows[0]["num_medium"]) == ("2", "0", "2")
