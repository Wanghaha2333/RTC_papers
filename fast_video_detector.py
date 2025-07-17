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


def process_video_fast(video_path: str) -> Tuple[str, bool, bool, str, str]:
    """处理单个视频的快速函数"""
    detector = FastVideoDetector()
    try:
        # 检查解码能力
        decodable, decode_info = detector.check_video_decodable(video_path)
        
        # 如果可解码，检查变化
        if decodable:
            no_change, change_info = detector.detect_no_change(video_path)
            has_change = not no_change
            return video_path, decodable, has_change, decode_info, change_info
        else:
            return video_path, decodable, False, decode_info, "跳过检测（无法解码）"
    except Exception as e:
        return video_path, False, False, f"错误: {str(e)}", "检测失败"


def batch_process_videos(video_paths: List[str], num_processes: int = None) -> List[Tuple[str, bool, bool, str, str]]:
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
    decodable_count = sum(1 for _, decodable, _, _, _ in results if decodable)
    has_change_count = sum(1 for _, _, has_change, _, _ in results if has_change)
    no_change_count = sum(1 for _, decodable, has_change, _, _ in results if decodable and not has_change)
    
    if not args.quiet:
        print(f"\n处理完成!")
        print(f"总数: {len(results)}")
        print(f"可解码: {decodable_count}")
        print(f"不可解码: {len(results) - decodable_count}")
        print(f"有变化: {has_change_count}")
        print(f"无变化: {no_change_count}")
        print(f"用时: {end_time - start_time:.2f} 秒")
        print(f"速度: {len(results) / (end_time - start_time):.1f} 个/秒")
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("视频文件,可解码,有变化,解码信息,变化信息\n")
            for path, decodable, has_change, decode_info, change_info in results:
                f.write(f'"{path}",{decodable},{has_change},"{decode_info}","{change_info}"\n')
        if not args.quiet:
            print(f"\n结果保存至: {args.output}")
    else:
        # 直接输出到终端
        for path, decodable, has_change, decode_info, change_info in results:
            if not decodable:
                status = "不可解码"
            elif has_change:
                status = "有变化"
            else:
                status = "无变化"
            print(f"{status}: {os.path.basename(path)}")


if __name__ == "__main__":
    main()