"""几个脚本共用的小工具函数。"""
from pathlib import Path

import yaml

# 仓库根目录 = 本文件所在目录(scripts/)的上一级
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(data_yaml, out_path=None) -> Path:
    """把数据集 yaml 里的相对 path 改成绝对路径, 另存一份并返回新文件路径。

    为什么需要: Ultralytics 遇到相对 path 时, 可能会去它自己的 datasets_dir 设置
    (通常是 ~/datasets)下找, 而不是相对于本仓库, 新手很容易在这里报 "dataset not found"。
    我们在 yaml 里写"相对仓库根目录"的路径(换电脑也能用), 运行前在这里统一转成绝对路径。
    """
    data_yaml = Path(data_yaml)
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    cfg["path"] = str(root)
    out_path = Path(out_path or REPO_ROOT / "runs" / "_tmp" / f"{data_yaml.stem}_resolved.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out_path
