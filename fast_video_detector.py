#!/usr/bin/env python3
"""
极速视频变化检测脚本
专为处理5万个视频优化，优先考虑速度

超级优化策略：
1. 最小化帧提取：只提取必要的4帧
2. 极小尺寸：缩放到64x48进行比较
3. 快速哈希：仅使用感知哈希进行比较
4. 内存池：重用内存空间
5. 多进程：充分利用多核CPU
"""

import cv2
import numpy as np
import argparse
import os
import sys
from typing import Tuple, List
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

class FastVideoDetector:
    def __init__(self, hash_threshold: int = 8):
        """
        极速视频检测器
        
        Args:
            hash_threshold: 感知哈希差异阈值（越小要求越严格）
        """
        self.hash_threshold = hash_threshold
        self.frame_size = (64, 48)  # 极小尺寸以提高速度
    
    def _compute_phash_fast(self, frame: np.ndarray) -> int:
        """超快速感知哈希计算"""
        # 直接缩放到8x8灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_NEAREST)
        
        # 计算平均值并生成哈希
        avg = np.mean(tiny)
        return sum(1 << i for i, pixel in enumerate(tiny.flat) if pixel > avg)
    
    def _hamming_distance(self, h1: int, h2: int) -> int:
        """汉明距离"""
        return bin(h1 ^ h2).count('1')
    
    def detect_no_change(self, video_path: str) -> Tuple[bool, str]:
        """
        超快速检测视频前4秒是否没有变化
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
                if not ret:
                    return False, f"无法读取第{second}秒帧"
                
                # 缩放并计算哈希
                small_frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_NEAREST)
                hash_val = self._compute_phash_fast(small_frame)
                hashes.append(hash_val)
            
            # 比较所有帧与第0秒帧的哈希差异
            base_hash = hashes[0]
            for i, h in enumerate(hashes[1:], 1):
                if self._hamming_distance(base_hash, h) > self.hash_threshold:
                    return False, f"第{i}秒有变化"
            
            return True, "无变化"
            
        finally:
            cap.release()


def process_video_fast(video_path: str) -> Tuple[str, bool, str]:
    """处理单个视频的快速函数"""
    detector = FastVideoDetector()
    try:
        no_change, info = detector.detect_no_change(video_path)
        return video_path, no_change, info
    except Exception as e:
        return video_path, False, f"错误: {str(e)}"


def batch_process_videos(video_paths: List[str], num_processes: int = None) -> List[Tuple[str, bool, str]]:
    """批量处理视频"""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_video_fast, video_paths))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="极速检测视频前4秒变化")
    parser.add_argument("input", help="视频文件或目录路径")
    parser.add_argument("-o", "--output", help="输出CSV文件路径")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="并行进程数")
    parser.add_argument("--hash-threshold", type=int, default=8, help="哈希差异阈值")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    # 收集视频文件
    video_paths = []
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    
    if os.path.isfile(args.input):
        video_paths = [args.input]
    elif os.path.isdir(args.input):
        for file_path in Path(args.input).iterdir():
            if file_path.suffix.lower() in video_exts:
                video_paths.append(str(file_path))
    else:
        print(f"错误: 路径不存在 {args.input}")
        sys.exit(1)
    
    if not video_paths:
        print("未找到视频文件")
        sys.exit(1)
    
    if not args.quiet:
        print(f"找到 {len(video_paths)} 个视频文件")
        print(f"使用 {args.jobs or multiprocessing.cpu_count()} 个进程")
    
    # 处理视频
    start_time = time.time()
    results = batch_process_videos(video_paths, args.jobs)
    end_time = time.time()
    
    # 统计结果
    no_change_count = sum(1 for _, no_change, _ in results if no_change)
    
    if not args.quiet:
        print(f"\n处理完成!")
        print(f"总数: {len(results)}")
        print(f"无变化: {no_change_count}")
        print(f"有变化: {len(results) - no_change_count}")
        print(f"用时: {end_time - start_time:.2f} 秒")
        print(f"速度: {len(results) / (end_time - start_time):.1f} 个/秒")
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("视频文件,无变化,详情\n")
            for path, no_change, info in results:
                f.write(f'"{path}",{no_change},"{info}"\n')
        if not args.quiet:
            print(f"\n结果保存至: {args.output}")
    else:
        # 直接输出到终端
        for path, no_change, info in results:
            status = "无变化" if no_change else "有变化"
            print(f"{status}: {os.path.basename(path)}")


if __name__ == "__main__":
    main()