#!/usr/bin/env bash
# 云服务器一键准备: 装依赖 -> 下载 BDD100K -> 解压 -> 整理训练集文件夹 -> 转换成 YOLO 格式
#
# 用法 (在仓库根目录运行, 先设置好 Kaggle token):
#   export KAGGLE_API_TOKEN=你的token
#   bash scripts/setup_cloud.sh
#
# 每一步都会先检查是否已经做过, 所以中途失败后重新运行是安全的。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_DIR=/workspace/data                      # 压缩包放在仓库外面, 不会被 git 跟踪
RAW="$REPO/datasets/bdd100k"                 # 解压位置, 和本地电脑上的目录结构保持一致
IMG="$RAW/bdd100k/bdd100k/images/100k"
LBL="$RAW/bdd100k_labels_release/bdd100k/labels"
OUT="$REPO/datasets/bdd100k_pedestrian"
cd "$REPO"

echo "== 1/5 安装依赖 (不动模板自带的 PyTorch)"
pip install -q ultralytics==8.3.139 pytest kaggle

echo "== 2/5 下载数据集"
if ! ls "$ZIP_DIR"/*.zip >/dev/null 2>&1; then
  : "${KAGGLE_API_TOKEN:?请先运行: export KAGGLE_API_TOKEN=你的token}"
  mkdir -p "$ZIP_DIR"
  kaggle datasets download solesensei/solesensei_bdd100k -p "$ZIP_DIR"
fi
ZIP="$(ls "$ZIP_DIR"/*.zip | head -1)"

echo "== 3/5 解压 (已存在的文件会跳过)"
mkdir -p "$RAW"
unzip -n -q "$ZIP" -d "$RAW"

echo "== 4/5 整理训练集: 这份 Kaggle 副本把训练图分进了子文件夹, 全部挪到 train/ 根目录"
for sub in trainA trainB testA testB; do
  if [ -d "$IMG/train/$sub" ]; then
    find "$IMG/train/$sub" -name '*.jpg' -exec mv -t "$IMG/train" {} +
    rmdir "$IMG/train/$sub"
  fi
done
echo "train 图片数: $(ls "$IMG/train" | wc -l)   (应为 70000)"
echo "val   图片数: $(ls "$IMG/val" | wc -l)   (应为 10000)"

echo "== 5/5 转换成 YOLO 格式 (用 scripts/bdd_core.py 里的转换函数)"
for split in train val; do
  if [ -f "$OUT/meta/${split}_meta.csv" ]; then
    echo "[$split] 已转换过, 跳过"
  else
    python scripts/convert_bdd_to_yolo.py \
      --labels "$LBL/bdd100k_labels_images_${split}.json" \
      --images-dir "$IMG/$split" --out-dir "$OUT" --split "$split"
  fi
done

pytest -q
echo "== 全部完成, 可以开始训练了"
