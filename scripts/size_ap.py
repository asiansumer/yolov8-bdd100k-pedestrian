"""按目标大小 (small/medium/large) 计算 AP50 和召回率的小工具 (纯 Python, 不依赖 ultralytics)。

为什么自己写: Ultralytics 的 val 只给整体 mAP, 不按大小拆分; COCO 官方工具 pycocotools
又需要把数据转成 COCO JSON, 对新手太绕。这里用最直接的方法:
  1. 每张图里, 预测框按置信度从高到低, 贪心地匹配 IoU >= 0.5 且还没被匹配过的真值框;
  2. 匹配上的预测 = TP, 归入"它所匹配真值框"的大小组;
     没匹配上的预测 = FP, 按"预测框自己的面积"归组 (与 COCO 的做法思路一致);
  3. 每个大小组内, 按置信度排序累计 TP/FP, 画 P-R 曲线, 用"全点插值"求面积 = AP50。

局限性 (面试时要能说清楚):
  - 匹配是在所有大小一起做的, 然后再分组, 和 COCO 的 areaRng 逻辑接近但不完全相同;
  - 用的是全点插值, COCO 用 101 点插值, Ultralytics 也有自己的插值, 数值会有小差异;
  - 所以这里的数字适合"本仓库内部 s/m/l 之间比较", 不要和别人论文里的数字逐位比较。
"""
SIZES = ("small", "medium", "large")


def iou(a, b) -> float:
    """两个 (x1, y1, x2, y2) 框的交并比。"""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def bucket_of(box) -> str:
    """COCO 面积阈值: < 32² 小, < 96² 中, 其余大 (原图像素)。"""
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def match_image(preds, gts, iou_thr=0.5):
    """单张图的匹配。preds: [(x1,y1,x2,y2,conf)], gts: [(x1,y1,x2,y2)]。

    返回 [(conf, is_tp, size)] 以及该图每个大小组的真值数量。
    """
    used = [False] * len(gts)
    records = []
    for p in sorted(preds, key=lambda x: -x[4]):  # 高置信度的先挑, 和标准 AP 计算一致
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if not used[j]:
                v = iou(p[:4], g)
                if v >= best_iou:
                    best_j, best_iou = j, v
        if best_j >= 0:
            used[best_j] = True
            records.append((p[4], True, bucket_of(gts[best_j])))
        else:
            records.append((p[4], False, bucket_of(p[:4])))
    n_gt = {s: sum(1 for g in gts if bucket_of(g) == s) for s in SIZES}
    return records, n_gt


def average_precision(records, n_gt: int):
    """records: [(conf, is_tp)]。返回 (AP, 最大召回率)。没有真值时返回 (None, None)。"""
    if n_gt == 0:
        return None, None
    tp = fp = 0
    recalls, precisions = [], []
    for _, is_tp in sorted(records, key=lambda r: -r[0]):
        tp += is_tp
        fp += not is_tp
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    if not recalls:
        return 0.0, 0.0
    # 精度"包络": 从右往左取最大值, 让 P-R 曲线单调不增 (标准做法)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, prev_r = 0.0, 0.0
    for r, p in zip(recalls, precisions):
        ap += (r - prev_r) * p  # 每一段召回率增量 × 对应精度 = 曲线下面积
        prev_r = r
    return ap, recalls[-1]


def size_ap(all_records, total_gt: dict) -> dict:
    """汇总所有图片的匹配结果, 每个大小组算一次 AP50。"""
    out = {}
    for s in SIZES:
        recs = [(c, t) for c, t, size in all_records if size == s]
        ap, rec = average_precision(recs, total_gt[s])
        out[s] = {"ap50": ap, "recall": rec, "num_gt": total_gt[s]}
    return out
