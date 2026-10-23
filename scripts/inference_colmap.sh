#!/usr/bin/env bash
# 在仓库根目录运行：bash scripts/inference_colmap.sh
# 修改场景路径、提示词和帧索引后运行；先激活已安装依赖的 Python 环境。
# 场景目录默认包含 images/ 和 sparse/0/（也支持模型直接放在 sparse/）。
# 索引按注册图像文件名排序，从 0 开始；目标区间至少包含 80 张图像。
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python inference_colmap.py \
  --scene "/data" \
  --reference-index 0 \
  --start 1 \
  --end 80 \
  --width 480 \
  --height 832 \
  --prompt "A quiet street filmed with a handheld camera." \
  --model_path "models/SCoPE" \
  --prepared-dir "outputs/input" \
  --output_path "outputs/video.mp4" \
  --seed 42

# 仅导出预处理结果：在命令末尾追加 --prepare-only。
# 显存卸载：按实际显存预算追加 --vram_limit_gb 40。
# 单位基线缩放：如需要，追加 --unit-baseline；默认保留平移尺度。
