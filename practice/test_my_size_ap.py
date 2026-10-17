"""my_size_ap.py 的分关测试。

只跑某一关 (在仓库根目录运行):  pytest practice -k level1 -v
全部跑一遍:                    pytest practice -v
"""
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import my_size_ap as mine  # noqa: E402  你写的版本
import size_ap as ref  # noqa: E402      仓库里原来的版本, 只用在最后的对比测试里


# ---------- 第 1 关: IoU ----------
def test_level1_identical_boxes():
    assert mine.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_level1_half_overlap():
    # 交集 5x10=50, 并集 100+100-50=150
    assert mine.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_level1_no_overlap():
    assert mine.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_level1_overlap_in_x_but_not_in_y():
    # x 方向重叠, y 方向不重叠: 宽度为正、高度为负, 交集必须是 0 而不是负数
    assert mine.iou((0, 0, 10, 10), (5, 20, 15, 30)) == 0.0


def test_level1_one_box_inside_another():
    # 小框完全在大框里: 交集 = 小框面积 25, 并集 = 大框面积 100
    assert mine.iou((0, 0, 10, 10), (2, 2, 7, 7)) == pytest.approx(25 / 100)


def test_level1_zero_area_boxes():
    assert mine.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------- 第 2 关: 贪心匹配 ----------
def test_level2_perfect_match():
    gts = [(0, 0, 20, 20), (100, 100, 200, 200)]  # 一个 small, 一个 large
    preds = [(*g, 0.9) for g in gts]
    records, n_gt = mine.match_image(preds, gts)
    assert sorted(records) == sorted([(0.9, True, "small"), (0.9, True, "large")])
    assert n_gt == {"small": 1, "medium": 0, "large": 1}


def test_level2_one_gt_matched_only_once():
    gts = [(0, 0, 20, 20)]
    preds = [(0, 0, 20, 20, 0.9), (1, 1, 20, 20, 0.8)]  # 重复框: 第二个应算 FP
    records, _ = mine.match_image(preds, gts)
    assert [r[1] for r in records] == [True, False]


def test_level2_higher_conf_goes_first():
    # 低分框排在列表前面, 但应该是高分框先挑真值
    gts = [(0, 0, 20, 20)]
    preds = [(1, 1, 20, 20, 0.3), (0, 0, 20, 20, 0.9)]
    records, _ = mine.match_image(preds, gts)
    assert records == [(0.9, True, "small"), (0.3, False, "small")]


def test_level2_picks_highest_iou_gt():
    # 第一个预测和两个真值的 IoU 都 >= 0.5 (0.6 和 0.905), 应该配给 IoU 更高的第二个真值。
    # 第二个预测只和第一个真值够得上 (0.667, 和第二个真值只有 0.333)。
    # 如果第一个预测错误地抢了第一个真值, 第二个预测就配不上了。
    gts = [(0, 0, 100, 100), (30, 0, 130, 100)]
    preds = [(25, 0, 125, 100, 0.9), (-20, 0, 80, 100, 0.8)]
    records, _ = mine.match_image(preds, gts)
    assert [r[1] for r in records] == [True, True]


def test_level2_below_threshold_is_fp():
    gts = [(0, 0, 10, 10)]
    preds = [(5, 0, 15, 10, 0.9)]  # IoU = 1/3 < 0.5
    records, _ = mine.match_image(preds, gts)
    assert records == [(0.9, False, "small")]


def test_level2_size_of_tp_and_fp():
    # TP 用真值框的大小, FP 用预测框自己的大小
    gts = [(0, 0, 40, 40)]  # 面积 1600 -> medium
    preds = [(0, 0, 40, 40, 0.9), (500, 500, 700, 700, 0.8)]  # 第二个是 200x200 的 FP -> large
    records, _ = mine.match_image(preds, gts)
    assert records == [(0.9, True, "medium"), (0.8, False, "large")]


# ---------- 第 3 关: P-R 点 ----------
EXAMPLE = [(0.95, True), (0.9, False), (0.8, True), (0.6, False), (0.5, True)]  # 讲义第 8 节的例子


def test_level3_example_from_notes():
    recalls, precisions = mine.pr_points(EXAMPLE, 4)
    assert recalls == pytest.approx([0.25, 0.25, 0.5, 0.5, 0.75])
    assert precisions == pytest.approx([1.0, 0.5, 2 / 3, 0.5, 0.6])


def test_level3_unsorted_input():
    shuffled = [EXAMPLE[i] for i in (3, 0, 4, 2, 1)]
    recalls, precisions = mine.pr_points(shuffled, 4)
    assert recalls == pytest.approx([0.25, 0.25, 0.5, 0.5, 0.75])
    assert precisions == pytest.approx([1.0, 0.5, 2 / 3, 0.5, 0.6])


# ---------- 第 4 关: 精度包络 ----------
def test_level4_example():
    assert mine.precision_envelope([1.0, 0.5, 2 / 3, 0.5, 0.6]) == pytest.approx(
        [1.0, 2 / 3, 2 / 3, 0.6, 0.6])


def test_level4_already_decreasing_unchanged():
    assert mine.precision_envelope([1.0, 0.8, 0.5]) == pytest.approx([1.0, 0.8, 0.5])


def test_level4_does_not_modify_input():
    p = [0.5, 1.0]
    mine.precision_envelope(p)
    assert p == [0.5, 1.0]


def test_level4_single_and_empty():
    assert mine.precision_envelope([0.3]) == pytest.approx([0.3])
    assert mine.precision_envelope([]) == []


# ---------- 第 5 关: 面积 ----------
def test_level5_example():
    recalls = [0.25, 0.25, 0.5, 0.5, 0.75]
    env = [1.0, 2 / 3, 2 / 3, 0.6, 0.6]
    # 0.25*1 + 0 + 0.25*(2/3) + 0 + 0.25*0.6
    assert mine.area_under_pr(recalls, env) == pytest.approx(0.25 + 0.25 * 2 / 3 + 0.25 * 0.6)


def test_level5_full_square():
    assert mine.area_under_pr([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0)


# ---------- 第 6 关: 组装 AP ----------
def test_level6_example_from_notes():
    ap, max_recall = mine.average_precision(EXAMPLE, 4)
    assert ap == pytest.approx(0.25 + 0.25 * 2 / 3 + 0.25 * 0.6)
    assert max_recall == pytest.approx(0.75)


def test_level6_false_positive_ranked_first_halves_precision():
    ap, max_recall = mine.average_precision([(0.95, False), (0.5, True)], 1)
    assert ap == pytest.approx(0.5)
    assert max_recall == pytest.approx(1.0)


def test_level6_special_cases():
    assert mine.average_precision([(0.9, False)], 0) == (None, None)
    assert mine.average_precision([], 3) == (0.0, 0.0)


# ---------- 第 7 关: 按大小汇总 ----------
def test_level7_perfect_predictions():
    gts = [(0, 0, 20, 20), (100, 100, 200, 200)]
    preds = [(*g, 0.9) for g in gts]
    records, n_gt = mine.match_image(preds, gts)
    res = mine.size_ap(records, n_gt)
    assert res["small"] == {"ap50": pytest.approx(1.0), "recall": pytest.approx(1.0), "num_gt": 1}
    assert res["large"]["ap50"] == pytest.approx(1.0)
    assert res["medium"] == {"ap50": None, "recall": None, "num_gt": 0}


# ---------- 毕业考试: 和原版在随机数据上对比 ----------
def _random_image(rng):
    gts, preds = [], []
    for _ in range(rng.randint(0, 6)):
        x, y = rng.uniform(0, 1000), rng.uniform(0, 600)
        w, h = rng.choice([rng.uniform(5, 30), rng.uniform(30, 90), rng.uniform(90, 300)]), rng.uniform(10, 300)
        gts.append((x, y, x + w, y + h))
    for g in gts:  # 大部分真值附近放一个带噪声的预测
        if rng.random() < 0.8:
            j = [rng.uniform(-8, 8) for _ in range(4)]
            preds.append((g[0] + j[0], g[1] + j[1], g[2] + j[2], g[3] + j[3], rng.random()))
    for _ in range(rng.randint(0, 3)):  # 再放几个纯误检
        x, y = rng.uniform(0, 1000), rng.uniform(0, 600)
        preds.append((x, y, x + rng.uniform(5, 200), y + rng.uniform(5, 200), rng.random()))
    return preds, gts


def test_level8_same_as_original_on_random_data():
    rng = random.Random(0)
    all_mine, all_ref = [], []
    gt_mine = {s: 0 for s in mine.SIZES}
    gt_ref = {s: 0 for s in ref.SIZES}
    for _ in range(300):
        preds, gts = _random_image(rng)
        r1, n1 = mine.match_image(preds, gts)
        r2, n2 = ref.match_image(preds, gts)
        all_mine += r1
        all_ref += r2
        for s in mine.SIZES:
            gt_mine[s] += n1[s]
            gt_ref[s] += n2[s]
    res_mine = mine.size_ap(all_mine, gt_mine)
    res_ref = ref.size_ap(all_ref, gt_ref)
    for s in mine.SIZES:
        assert res_mine[s]["num_gt"] == res_ref[s]["num_gt"]
        assert res_mine[s]["ap50"] == pytest.approx(res_ref[s]["ap50"])
        assert res_mine[s]["recall"] == pytest.approx(res_ref[s]["recall"])
