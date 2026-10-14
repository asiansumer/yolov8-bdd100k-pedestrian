"""按大小算 AP 的匹配器的单元测试。运行: pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import size_ap  # noqa: E402


def test_iou():
    assert size_ap.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert size_ap.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert size_ap.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_perfect_predictions_give_ap_1():
    gts = [(0, 0, 20, 20), (100, 100, 200, 200)]  # 一个 small, 一个 large
    preds = [(*g, 0.9) for g in gts]
    records, n_gt = size_ap.match_image(preds, gts)
    res = size_ap.size_ap(records, n_gt)
    assert res["small"]["ap50"] == pytest.approx(1.0)
    assert res["large"]["ap50"] == pytest.approx(1.0)
    assert res["medium"]["ap50"] is None  # 没有中目标真值 -> 不计算


def test_false_positive_ranked_first_halves_precision():
    gts = [(0, 0, 20, 20)]
    preds = [(500, 500, 515, 515, 0.95), (0, 0, 20, 20, 0.5)]  # 高分误检 + 低分正确
    records, n_gt = size_ap.match_image(preds, gts)
    res = size_ap.size_ap(records, n_gt)
    assert res["small"]["ap50"] == pytest.approx(0.5)  # 召回到 1 时精度是 1/2
    assert res["small"]["recall"] == pytest.approx(1.0)


def test_one_gt_matched_only_once():
    gts = [(0, 0, 20, 20)]
    preds = [(0, 0, 20, 20, 0.9), (1, 1, 20, 20, 0.8)]  # 重复框: 第二个应算 FP
    records, _ = size_ap.match_image(preds, gts)
    assert [r[1] for r in records] == [True, False]
