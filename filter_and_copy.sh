#!/bin/bash

# 视频筛选和复制批处理脚本
# 使用方法: ./filter_and_copy.sh [输入目录] [输出目录] [进程数]

set -e

# 默认参数
INPUT_DIR="${1:-./videos}"
OUTPUT_DIR="${2:-./filtered_videos}"
JOBS="${3:-$(nproc)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_FILE="filter_report_$(date +%Y%m%d_%H%M%S).txt"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 视频筛选和复制脚本 ===${NC}"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "并行进程: $JOBS"
echo "报告文件: $REPORT_FILE"
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
FILTER_SCRIPT="$SCRIPT_DIR/video_filter_copy.py"
if [ ! -f "$FILTER_SCRIPT" ]; then
    echo -e "${RED}错误: 未找到脚本文件: $FILTER_SCRIPT${NC}"
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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始处理
echo -e "${YELLOW}开始分析和复制视频...${NC}"
echo -e "${BLUE}筛选条件: 可正确解码 + 前4秒有变化${NC}"
echo

START_TIME=$(date +%s)

python3 "$FILTER_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" \
    -j "$JOBS" \
    --hash-threshold 8 \
    --report "$REPORT_FILE"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 统计最终结果
if [ -f "$REPORT_FILE" ]; then
    echo
    echo -e "${GREEN}=== 处理完成 ===${NC}"
    echo "总处理时间: ${DURATION} 秒"
    
    # 从报告文件中提取统计信息
    TOTAL_VIDEOS=$(grep "总视频数:" "$REPORT_FILE" | cut -d':' -f2 | tr -d ' ')
    DECODABLE=$(grep "可解码视频:" "$REPORT_FILE" | cut -d':' -f2 | tr -d ' ')
    HAS_CHANGE=$(grep "有变化视频:" "$REPORT_FILE" | cut -d':' -f2 | tr -d ' ')
    COPIED=$(grep "复制视频数:" "$REPORT_FILE" | cut -d':' -f2 | tr -d ' ')
    
    echo "总视频数: $TOTAL_VIDEOS"
    echo -e "${GREEN}可解码: $DECODABLE${NC}"
    echo -e "${YELLOW}有变化: $HAS_CHANGE${NC}"
    echo -e "${BLUE}已复制: $COPIED${NC}"
    echo "处理速度: $(python3 -c "print(f'{$TOTAL_VIDEOS / max($DURATION, 1):.1f}')")个/秒"
    
    # 计算统计百分比
    if [ "$TOTAL_VIDEOS" -gt 0 ]; then
        DECODE_RATE=$(python3 -c "print(f'{$DECODABLE / $TOTAL_VIDEOS * 100:.1f}')")
        CHANGE_RATE=$(python3 -c "print(f'{$HAS_CHANGE / $TOTAL_VIDEOS * 100:.1f}')")
        COPY_RATE=$(python3 -c "print(f'{$COPIED / $TOTAL_VIDEOS * 100:.1f}')")
        
        echo
        echo "统计比例:"
        echo "解码成功率: ${DECODE_RATE}%"
        echo "变化检出率: ${CHANGE_RATE}%"
        echo "最终筛选率: ${COPY_RATE}%"
    fi
    
    echo
    echo "输出目录: $OUTPUT_DIR"
    echo "详细报告: $REPORT_FILE"
    
    # 显示输出目录的文件数量
    COPIED_FILES=$(find "$OUTPUT_DIR" -type f | wc -l)
    echo "输出目录文件数: $COPIED_FILES"
    
    # 生成简单总结
    echo
    echo -e "${GREEN}=== 总结 ===${NC}"
    if [ "$COPIED" -gt 0 ]; then
        echo -e "${GREEN}✓ 成功筛选出 $COPIED 个满足条件的视频${NC}"
        echo -e "${BLUE}✓ 这些视频都可以正确解码且前4秒有变化${NC}"
    else
        echo -e "${YELLOW}⚠ 没有找到满足条件的视频${NC}"
        echo -e "${YELLOW}  所有视频要么无法解码，要么前4秒没有变化${NC}"
    fi
    
    # 如果有不可解码的视频，显示提示
    NOT_DECODABLE=$((TOTAL_VIDEOS - DECODABLE))
    if [ "$NOT_DECODABLE" -gt 0 ]; then
        echo -e "${RED}⚠ 发现 $NOT_DECODABLE 个无法解码的视频，请检查报告文件${NC}"
    fi
    
else
    echo -e "${RED}错误: 处理失败，未生成报告文件${NC}"
    exit 1
fi

echo
echo -e "${GREEN}脚本执行完成！${NC}"