
SIZES = ("small", "medium", "large")


def bucket_of(box) -> str:
    """(已给出) 按 COCO 面积阈值分组: 面积 < 32² 为小, < 96² 为中, 其余为大 (原图像素)。"""
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"

def iou(a, b) -> float:
    """两个 (x1, y1, x2, y2) 框的交并比, 返回 0 到 1 之间的小数。
    """
    # 交集框inter: 起点取 max, 终点取 min
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    inter_width = max(inter[2] - inter[0], 0)
    inter_height = max(inter[3] - inter[1], 0)
    s_inter = inter_width * inter_height
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])
    s_union = s_a + s_b - s_inter
    if s_union == 0:
        return 0.0
    return s_inter / s_union

def match_image(preds, gts, iou_thr=0.5):
    """把一张图里的预测框和真值框配对。"""

    used = [False] * len(gts)
    records = []

    # 把 preds 排序。key=lambda p: p[4] 的意思是“按每个框的第 5 个数（conf）来排”，reverse=True 表示从大到小。
    preds_sorted = sorted(preds, key=lambda p: p[4], reverse=True)

    for p in preds_sorted:
        best_j = -1
        best_v = 0
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(p[:4], g)
            if v > best_v:
                best_v = v
                best_j = j

        if best_v >= iou_thr:
            used[best_j] = True
            records.append((p[4], True, bucket_of(gts[best_j])))
        else:
            records.append((p[4], False, bucket_of(p[:4])))


    n_gt = {"small":0, "medium":0, "large":0}

    for g in gts:
        size = bucket_of(g)
        n_gt[size] += 1

    return records, n_gt


def pr_points(records, n_gt: int):
    """records: [(conf, is_tp), ...] (注意这里只有两项, 没有 size)
    n_gt: 这一组真值框的总数
    返回 (recalls, precisions) 两个等长列表:
      按 conf 从高到低, 每"多接受一个预测", 就把当前的 recall 和 precision 各记一次。
    """

    tp, fp = 0, 0
    recalls = []
    precisions = []

    for _, is_tp in sorted(records, key=lambda r: r[0], reverse=True):
        if is_tp:
            tp += 1
        else:
            fp += 1

        recall = tp / n_gt
        precision = tp / (tp + fp)
        recalls.append(recall)
        precisions.append(precision)

    return recalls, precisions



def precision_envelope(precisions):
    """返回一个新列表: 每个位置换成"它自己以及它右边所有值里的最大值"。
    """

    out = list(precisions)
    for i in range(len(precisions) - 2, -1, -1):
        out[i] = max(out[i], out[i+1])

    return out


def area_under_pr(recalls, precisions) -> float:
    """全点插值的面积: 把曲线切成一个个竖长条, 每条的面积 = 宽 × 高。
      宽 = 这一点的 recall - 上一点的 recall (第一个点的"上一点"当作 recall = 0)
      高 = 这一点的 precision (已经做过包络的)
    """

    ap = 0
    prev_r = 0

    for r, p in zip(recalls, precisions):
        ap += (r - prev_r) * p
        prev_r = r

    return ap


def average_precision(records, n_gt: int):
    """records: [(conf, is_tp), ...]。返回 (AP, 最大召回率)。
    """

    if n_gt == 0:
        return None, None
    if not records:
        return 0.0, 0.0

    recalls,precisions = pr_points(records, n_gt)
    env = precision_envelope(precisions)
    ap = area_under_pr(recalls, env)
    return ap,recalls[-1]


def size_ap(all_records, total_gt: dict) -> dict:
    """all_records: 所有图片的 records 拼在一起, [(conf, is_tp, size), ...]
    total_gt: {"small": 总数, "medium": 总数, "large": 总数}
    """
    out = {}
    for s in SIZES:
        recs = []
        for conf, is_tp, size in all_records:
            if size == s:
                recs.append((conf,is_tp))
        ap, rec = average_precision(recs, total_gt[s])
        out[s] = {"ap50": ap, "recall": rec, "num_gt": total_gt[s]}

    return out
