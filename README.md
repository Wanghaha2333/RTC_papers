# 视频变化检测与筛选工具

专为大规模批量处理设计的高效视频处理工具集，支持解码检测、变化分析和智能筛选复制功能。

## 🚀 核心功能

- **解码检测**: 自动识别无法正确解码的损坏视频
- **变化分析**: 快速判断视频前4秒（第0、1、2、3秒）是否有变化
- **智能筛选**: 自动复制满足条件的视频到新文件夹
- **批量处理**: 针对5万个视频优化，使用感知哈希算法
- **多进程并行**: 充分利用多核CPU
- **内存优化**: 低内存占用，适合大批量处理

## 📦 文件结构

```
├── video_filter_copy.py        # 筛选复制脚本（主要功能）
├── fast_video_detector.py      # 快速分析脚本
├── video_change_detector.py    # 精确检测脚本
├── filter_and_copy.sh          # 筛选复制批处理脚本
├── batch_process.sh            # 分析批处理脚本
├── requirements.txt            # 依赖文件
├── 使用说明.md                 # 详细使用说明
└── README.md                   # 本文件
```

## 🛠️ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 🎯 筛选复制视频（主要功能）

```bash
# 筛选出可解码且有变化的视频到新文件夹
./filter_and_copy.sh /path/to/input_videos/ /path/to/output_videos/

# 或直接使用Python脚本
python3 video_filter_copy.py /path/to/input_videos/ /path/to/output_videos/
```

### 3. 📊 快速分析视频

```bash
# 分析视频状态，生成报告
python3 fast_video_detector.py /path/to/videos/ -o analysis_results.csv
```

### 4. 🔧 使用批处理脚本

```bash
# 筛选复制
./filter_and_copy.sh /path/to/videos/ /path/to/output/ 8

# 快速分析
./batch_process.sh /path/to/videos/ results.csv 8
```

## ⚡ 性能

- **处理速度**: 10-50个视频/秒（取决于硬件和视频大小）
- **内存占用**: 极低，适合大批量处理
- **预计时间**: 处理5万个视频约20分钟-2小时

## 📊 输出格式

### 筛选复制模式
- **输出**: 筛选后的视频文件 + 详细报告
- **筛选条件**: 可正确解码 + 前4秒有变化

### 分析模式
结果以CSV格式保存：

```csv
视频文件,可解码,有变化,解码信息,变化信息
"/path/video1.mp4",True,True,"解码正常,时长10.5s,30.0fps","第2秒有变化"
"/path/video2.mp4",False,False,"无法打开视频文件","跳过检测（无法解码）"
"/path/video3.mp4",True,False,"解码正常,时长8.2s,25.0fps","前4秒无变化"
```

## 🔧 主要参数

- `-j`: 并行进程数（默认为CPU核心数）
- `--hash-threshold`: 检测敏感度（默认8，越小越严格）
- `-o`: 输出文件路径
- `-q`: 静默模式

## 📖 详细文档

查看 [使用说明.md](使用说明.md) 获取完整的使用指南和优化建议。

## 🎯 算法原理

1. **感知哈希**: 将帧缩放到8x8像素，生成64位哈希值
2. **汉明距离**: 比较哈希值的位差异
3. **阈值判断**: 超过阈值认为有变化

## 📝 示例输出

```
找到 1000 个视频文件
使用 8 个进程

处理完成!
总数: 1000
无变化: 156
有变化: 844
用时: 45.67 秒
速度: 21.9 个/秒
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📄 许可证

MIT License
