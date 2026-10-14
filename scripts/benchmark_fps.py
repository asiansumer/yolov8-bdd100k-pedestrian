"""测量模型推理速度 (FPS)。

为什么要先"预热": 前几次推理会触发 MPS/GPU 的内核编译、内存分配等一次性开销,
如果算进去, FPS 会被严重低估。所以先空跑若干张, 再正式计时。

注意: FPS 强烈依赖硬件、batch 大小和是否包含前后处理。论文里的 FPS 和这里在 M3 Max 上
测到的数不能直接比较, 只能在同一台机器上比较 s/m/l 之间的相对快慢。

用法:
  python scripts/benchmark_fps.py --weights runs/v8s_e30/weights/best.pt --n 200
"""
import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from common import REPO_ROOT


def sync(device: str):
    """等 GPU/MPS 上排队的计算真正做完再计时, 否则测到的只是"提交任务"的时间。"""
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device not in ("cpu", "mps") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description="推理速度测试")
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--images-dir", default=REPO_ROOT / "datasets" / "bdd100k_pedestrian" / "images" / "val",
                   type=Path)
    p.add_argument("--n", type=int, default=200, help="计时用的图片数")
    p.add_argument("--warmup", type=int, default=10, help="预热的图片数 (不计时)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    images = sorted(args.images_dir.glob("*.jpg"))[: args.warmup + args.n]
    if len(images) <= args.warmup:
        raise SystemExit(f"{args.images_dir} 里的图片不够 (需要 > {args.warmup} 张)")
    model = YOLO(str(args.weights))
    kw = dict(imgsz=args.imgsz, device=args.device, verbose=False)

    for im in images[: args.warmup]:
        model.predict(str(im), **kw)

    timed = images[args.warmup:]
    speeds = []  # Ultralytics 自带的分段耗时 (毫秒): 预处理 / 推理 / 后处理(NMS)
    sync(args.device)
    t0 = time.perf_counter()
    for im in timed:
        r = model.predict(str(im), **kw)[0]
        speeds.append(r.speed)
    sync(args.device)
    total = time.perf_counter() - t0

    # 端到端 FPS 包含读图、预处理、推理、NMS, 是最贴近实际部署的数字
    print(f"模型: {args.weights}  设备: {args.device}  图片数: {len(timed)}  batch=1")
    print(f"端到端 FPS: {len(timed) / total:.1f}  (平均 {total / len(timed) * 1000:.1f} ms/张)")
    for k in ("preprocess", "inference", "postprocess"):
        print(f"  平均 {k}: {sum(s[k] for s in speeds) / len(speeds):.1f} ms")


if __name__ == "__main__":
    main()
