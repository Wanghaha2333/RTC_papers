#!/usr/bin/env python3
"""
视频筛选和复制脚本
功能：
1. 检测视频是否能正确解码
2. 检测前4秒（0-3秒）是否有变化
3. 将同时满足"可解码"和"有变化"的视频复制到输出文件夹

优化策略：
- 快速解码检测
- 感知哈希变化检测
- 多进程并行处理
- 文件复制优化
"""

import cv2
import numpy as np
import argparse
import os
import sys
import shutil
from typing import Tuple, List, Optional
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

class VideoProcessor:
    def __init__(self, hash_threshold: int = 8):
        """
        视频处理器
        
        Args:
            hash_threshold: 感知哈希差异阈值（越小要求越严格）
        """
        self.hash_threshold = hash_threshold
        self.frame_size = (64, 48)  # 小尺寸以提高速度

    def check_video_decodable(self, video_path: str) -> Tuple[bool, str]:
        """
        检查视频是否可以正确解码
        
        Returns:
            (是否可解码, 详细信息)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "无法打开视频文件"
            
            # 检查基本属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                cap.release()
                return False, "视频属性异常"
            
            # 尝试读取前几帧来验证解码能力
            successful_reads = 0
            test_frames = min(10, int(frame_count))  # 测试前10帧或总帧数
            
            for i in range(test_frames):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    successful_reads += 1
                else:
                    break
            
            cap.release()
            
            # 如果成功读取的帧数少于测试帧数的80%，认为解码有问题
            success_rate = successful_reads / test_frames
            if success_rate < 0.8:
                return False, f"解码成功率低: {success_rate:.1%}"
            
            # 检查视频长度是否足够（至少4秒）
            duration = frame_count / fps
            if duration < 4.0:
                return False, f"视频时长不足4秒: {duration:.1f}s"
            
            return True, f"解码正常,时长{duration:.1f}s,{fps:.1f}fps"
            
        except Exception as e:
            return False, f"解码检测错误: {str(e)}"

    def _compute_phash_fast(self, frame: np.ndarray) -> int:
        """超快速感知哈希计算"""
        try:
            # 直接缩放到8x8灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tiny = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_NEAREST)
            
            # 计算平均值并生成哈希
            avg = np.mean(tiny)
            return sum(1 << i for i, pixel in enumerate(tiny.flat) if pixel > avg)
        except:
            return 0
    
    def _hamming_distance(self, h1: int, h2: int) -> int:
        """汉明距离"""
        return bin(h1 ^ h2).count('1')
    
    def check_video_has_change(self, video_path: str) -> Tuple[bool, str]:
        """
        检测视频前4秒是否有变化
        
        Returns:
            (是否有变化, 详细信息)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "无法打开视频"
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return False, "无效的FPS"
            
            # 提取4个关键帧的哈希值
            hashes = []
            for second in [0, 1, 2, 3]:
                frame_num = int(second * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    return False, f"无法读取第{second}秒帧"
                
                # 缩放并计算哈希
                small_frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_NEAREST)
                hash_val = self._compute_phash_fast(small_frame)
                hashes.append(hash_val)
            
            # 比较所有帧与第0秒帧的哈希差异
            base_hash = hashes[0]
            for i, h in enumerate(hashes[1:], 1):
                if self._hamming_distance(base_hash, h) > self.hash_threshold:
                    return True, f"第{i}秒有变化"
            
            return False, "前4秒无变化"
            
        except Exception as e:
            return False, f"变化检测错误: {str(e)}"
        finally:
            cap.release()

    def process_video(self, video_path: str) -> dict:
        """
        处理单个视频
        
        Returns:
            处理结果字典
        """
        result = {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'decodable': False,
            'has_change': False,
            'should_copy': False,
            'decode_info': '',
            'change_info': ''
        }
        
        # 1. 检查解码能力
        decodable, decode_info = self.check_video_decodable(video_path)
        result['decodable'] = decodable
        result['decode_info'] = decode_info
        
        # 2. 如果可解码，检查是否有变化
        if decodable:
            has_change, change_info = self.check_video_has_change(video_path)
            result['has_change'] = has_change
            result['change_info'] = change_info
            
            # 3. 判断是否应该复制（可解码且有变化）
            result['should_copy'] = decodable and has_change
        else:
            result['change_info'] = '跳过检测（无法解码）'
        
        return result


def process_single_video(args: Tuple[str, int]) -> dict:
    """处理单个视频的包装函数，用于多进程"""
    video_path, hash_threshold = args
    processor = VideoProcessor(hash_threshold)
    return processor.process_video(video_path)


def copy_video_file(src_path: str, dest_dir: str, preserve_structure: bool = False, base_dir: str = "") -> bool:
    """
    复制视频文件到目标目录
    
    Args:
        src_path: 源文件路径
        dest_dir: 目标目录
        preserve_structure: 是否保持目录结构
        base_dir: 基础目录（用于计算相对路径）
    
    Returns:
        是否复制成功
    """
    try:
        if preserve_structure and base_dir:
            # 保持目录结构
            rel_path = os.path.relpath(src_path, base_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            dest_parent = os.path.dirname(dest_path)
            os.makedirs(dest_parent, exist_ok=True)
        else:
            # 直接复制到目标目录
            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, filename)
        
        shutil.copy2(src_path, dest_path)
        return True
    except Exception as e:
        print(f"复制失败 {src_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="视频筛选和复制工具")
    parser.add_argument("input_dir", help="输入视频目录")
    parser.add_argument("output_dir", help="输出目录（复制满足条件的视频）")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="并行进程数")
    parser.add_argument("--hash-threshold", type=int, default=8, help="哈希差异阈值")
    parser.add_argument("--preserve-structure", action="store_true", help="保持目录结构")
    parser.add_argument("--report", help="生成详细报告文件路径")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("--extensions", default="mp4,avi,mov,mkv,flv,wmv,m4v,webm", 
                       help="视频文件扩展名")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集视频文件
    video_paths = []
    video_exts = {f'.{ext.strip().lower()}' for ext in args.extensions.split(',')}
    
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if Path(file).suffix.lower() in video_exts:
                video_paths.append(os.path.join(root, file))
    
    if not video_paths:
        print("未找到视频文件")
        sys.exit(1)
    
    if not args.quiet:
        print(f"找到 {len(video_paths)} 个视频文件")
        print(f"使用 {args.jobs or multiprocessing.cpu_count()} 个进程")
        print(f"输出目录: {args.output_dir}")
        print()
    
    # 处理视频
    start_time = time.time()
    
    # 准备多进程参数
    process_args = [(path, args.hash_threshold) for path in video_paths]
    
    # 并行处理
    if args.jobs is None:
        args.jobs = multiprocessing.cpu_count()
    
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        if not args.quiet:
            print("正在分析视频...")
        results = list(executor.map(process_single_video, process_args))
    
    analysis_time = time.time()
    
    # 统计和复制
    decodable_count = sum(1 for r in results if r['decodable'])
    has_change_count = sum(1 for r in results if r['has_change'])
    copy_count = sum(1 for r in results if r['should_copy'])
    
    if not args.quiet:
        print(f"\n分析完成! 用时: {analysis_time - start_time:.2f} 秒")
        print(f"总视频数: {len(results)}")
        print(f"可解码: {decodable_count}")
        print(f"有变化: {has_change_count}")
        print(f"需复制: {copy_count}")
        print()
    
    # 复制满足条件的视频
    if copy_count > 0:
        if not args.quiet:
            print("开始复制视频...")
        
        copied_count = 0
        for result in results:
            if result['should_copy']:
                success = copy_video_file(
                    result['path'], 
                    args.output_dir, 
                    args.preserve_structure,
                    args.input_dir
                )
                if success:
                    copied_count += 1
                    if not args.quiet and copied_count % 100 == 0:
                        print(f"已复制 {copied_count}/{copy_count} 个视频")
        
        copy_time = time.time()
        
        if not args.quiet:
            print(f"\n复制完成! 用时: {copy_time - analysis_time:.2f} 秒")
            print(f"成功复制: {copied_count}/{copy_count} 个视频")
    else:
        if not args.quiet:
            print("没有满足条件的视频需要复制")
    
    total_time = time.time() - start_time
    
    # 生成报告
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("视频筛选和复制报告\n")
            f.write("==================\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {args.input_dir}\n")
            f.write(f"输出目录: {args.output_dir}\n")
            f.write(f"总处理时间: {total_time:.2f} 秒\n")
            f.write(f"总视频数: {len(results)}\n")
            f.write(f"可解码视频: {decodable_count}\n")
            f.write(f"有变化视频: {has_change_count}\n")
            f.write(f"复制视频数: {copied_count if copy_count > 0 else 0}\n")
            f.write(f"处理速度: {len(results) / total_time:.1f} 个/秒\n\n")
            
            # 详细结果
            f.write("详细结果:\n")
            f.write("文件路径,可解码,解码信息,有变化,变化信息,已复制\n")
            for result in results:
                f.write(f'"{result["path"]}",{result["decodable"]},"{result["decode_info"]}",'
                       f'{result["has_change"]},"{result["change_info"]}",{result["should_copy"]}\n')
            
            # 分类列表
            f.write("\n\n=== 不可解码的视频 ===\n")
            for result in results:
                if not result['decodable']:
                    f.write(f"{result['path']} - {result['decode_info']}\n")
            
            f.write("\n=== 可解码但无变化的视频 ===\n")
            for result in results:
                if result['decodable'] and not result['has_change']:
                    f.write(f"{result['path']} - {result['change_info']}\n")
            
            f.write("\n=== 已复制的视频（可解码且有变化）===\n")
            for result in results:
                if result['should_copy']:
                    f.write(f"{result['path']} - {result['change_info']}\n")
        
        if not args.quiet:
            print(f"详细报告已保存到: {args.report}")
    
    # 最终统计
    if not args.quiet:
        print(f"\n=== 最终统计 ===")
        print(f"总处理时间: {total_time:.2f} 秒")
        print(f"处理速度: {len(results) / total_time:.1f} 个/秒")
        print(f"总视频: {len(results)}")
        print(f"不可解码: {len(results) - decodable_count}")
        print(f"可解码但无变化: {decodable_count - has_change_count}")
        print(f"可解码且有变化(已复制): {copied_count if copy_count > 0 else 0}")


if __name__ == "__main__":
    main()