#!/bin/bash

# 批量视频检测脚本
# 使用方法: ./batch_process.sh [视频目录] [输出文件] [进程数]

set -e

# 默认参数
INPUT_DIR="${1:-./videos}"
OUTPUT_FILE="${2:-results_$(date +%Y%m%d_%H%M%S).csv}"
JOBS="${3:-$(nproc)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 视频变化检测批处理脚本 ===${NC}"
echo "输入目录: $INPUT_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "并行进程: $JOBS"
echo

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 检查Python和依赖
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3${NC}"
    exit 1
fi

# 检查脚本文件
FAST_SCRIPT="$SCRIPT_DIR/fast_video_detector.py"
if [ ! -f "$FAST_SCRIPT" ]; then
    echo -e "${RED}错误: 未找到脚本文件: $FAST_SCRIPT${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
python3 -c "import cv2, numpy" 2>/dev/null || {
    echo -e "${RED}错误: 缺少依赖，请运行: pip install -r requirements.txt${NC}"
    exit 1
}

# 统计视频文件数量
echo -e "${YELLOW}统计视频文件...${NC}"
VIDEO_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.m4v" -o -iname "*.webm" \) | wc -l)

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo -e "${RED}错误: 在目录中未找到视频文件${NC}"
    exit 1
fi

echo -e "${GREEN}找到 $VIDEO_COUNT 个视频文件${NC}"

# 开始处理
echo -e "${YELLOW}开始处理...${NC}"
START_TIME=$(date +%s)

python3 "$FAST_SCRIPT" "$INPUT_DIR" \
    -o "$OUTPUT_FILE" \
    -j "$JOBS" \
    --hash-threshold 8

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 统计结果
if [ -f "$OUTPUT_FILE" ]; then
    NO_CHANGE_COUNT=$(grep ',True,' "$OUTPUT_FILE" | wc -l)
    HAS_CHANGE_COUNT=$(grep ',False,' "$OUTPUT_FILE" | wc -l)
    
    echo
    echo -e "${GREEN}=== 处理完成 ===${NC}"
    echo "总处理时间: ${DURATION} 秒"
    echo "处理速度: $(python3 -c "print(f'{$VIDEO_COUNT / max($DURATION, 1):.1f}')")个/秒"
    echo "总视频数: $VIDEO_COUNT"
    echo -e "${GREEN}无变化: $NO_CHANGE_COUNT${NC}"
    echo -e "${YELLOW}有变化: $HAS_CHANGE_COUNT${NC}"
    echo "结果文件: $OUTPUT_FILE"
    
    # 生成简单报告
    REPORT_FILE="${OUTPUT_FILE%.*}_report.txt"
    {
        echo "视频变化检测报告"
        echo "=================="
        echo "处理时间: $(date)"
        echo "输入目录: $INPUT_DIR"
        echo "总视频数: $VIDEO_COUNT"
        echo "无变化: $NO_CHANGE_COUNT"
        echo "有变化: $HAS_CHANGE_COUNT"
        echo "处理时间: ${DURATION}秒"
        echo "处理速度: $(python3 -c "print(f'{$VIDEO_COUNT / max($DURATION, 1):.1f}')")个/秒"
        echo
        echo "无变化的视频文件："
        grep ',True,' "$OUTPUT_FILE" | cut -d',' -f1 | tr -d '"'
    } > "$REPORT_FILE"
    
    echo "报告文件: $REPORT_FILE"
else
    echo -e "${RED}错误: 输出文件未生成${NC}"
    exit 1
fi