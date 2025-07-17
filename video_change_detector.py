#!/usr/bin/env python3
"""
高效视频变化检测脚本
检测视频前4秒（第0、1、2、3秒）是否完全没有变化

优化策略：
1. 降采样：提取关键帧而非逐帧分析
2. 图像降尺寸：减少计算量
3. 多种比较算法：结构相似性、直方图、哈希等
4. 早期退出：一旦检测到变化立即返回
5. 内存优化：及时释放帧数据
"""

import cv2
import numpy as np
import argparse
import os
import sys
from typing import Tuple, Optional, List
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

class VideoChangeDetector:
    def __init__(self, 
                 resize_width: int = 160,
                 resize_height: int = 120,
                 ssim_threshold: float = 0.98,
                 hist_threshold: float = 0.99,
                 hash_threshold: int = 5):
        """
        初始化视频变化检测器
        
        Args:
            resize_width: 缩放后的宽度（越小速度越快）
            resize_height: 缩放后的高度
            ssim_threshold: SSIM相似度阈值（越高要求越严格）
            hist_threshold: 直方图相似度阈值
            hash_threshold: 感知哈希差异阈值
        """
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.hash_threshold = hash_threshold
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """快速缩放帧以减少计算量"""
        return cv2.resize(frame, (self.resize_width, self.resize_height), 
                         interpolation=cv2.INTER_LINEAR)
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算结构相似性指数（SSIM）"""
        # 转换为灰度图
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 计算均值
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return np.mean(ssim_map)
    
    def _compute_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算直方图相似度"""
        # 转换为HSV颜色空间
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # 计算直方图
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # 计算相关性
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def _compute_phash(self, img: np.ndarray) -> int:
        """计算感知哈希"""
        # 转换为灰度图并缩放到8x8
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_LINEAR)
        
        # 计算平均值
        avg = np.mean(resized)
        
        # 生成哈希
        hash_val = 0
        for i in range(8):
            for j in range(8):
                if resized[i, j] > avg:
                    hash_val |= (1 << (i * 8 + j))
        
        return hash_val
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        return bin(hash1 ^ hash2).count('1')
    
    def _frames_are_identical(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        判断两帧是否相同
        使用多种算法确保准确性，按计算复杂度从低到高排序
        """
        # 1. 感知哈希比较（最快）
        hash1 = self._compute_phash(frame1)
        hash2 = self._compute_phash(frame2)
        if self._hamming_distance(hash1, hash2) > self.hash_threshold:
            return False
        
        # 2. 直方图比较（中等速度）
        hist_sim = self._compute_histogram_similarity(frame1, frame2)
        if hist_sim < self.hist_threshold:
            return False
        
        # 3. SSIM比较（较慢但最准确）
        ssim = self._compute_ssim(frame1, frame2)
        return ssim >= self.ssim_threshold
    
    def extract_frames_at_seconds(self, video_path: str, seconds: List[int]) -> Optional[List[np.ndarray]]:
        """
        提取指定秒数的帧
        
        Args:
            video_path: 视频文件路径
            seconds: 要提取的秒数列表
            
        Returns:
            提取的帧列表，如果失败返回None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return None
            
            frames = []
            for second in seconds:
                # 跳转到指定秒数
                frame_number = int(second * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    return None
                
                # 缩放帧以提高处理速度
                resized_frame = self._resize_frame(frame)
                frames.append(resized_frame)
            
            return frames
            
        finally:
            cap.release()
    
    def detect_no_change(self, video_path: str) -> Tuple[bool, str]:
        """
        检测视频前4秒是否完全没有变化
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (是否没有变化, 详细信息)
        """
        if not os.path.exists(video_path):
            return False, f"文件不存在: {video_path}"
        
        # 提取第0、1、2、3秒的帧
        frames = self.extract_frames_at_seconds(video_path, [0, 1, 2, 3])
        if frames is None or len(frames) != 4:
            return False, f"无法提取帧: {video_path}"
        
        # 检查相邻帧是否相同
        for i in range(1, 4):
            if not self._frames_are_identical(frames[0], frames[i]):
                return False, f"第0秒和第{i}秒帧不同"
        
        return True, "前4秒完全没有变化"


def process_single_video(args: Tuple[str, VideoChangeDetector]) -> Tuple[str, bool, str]:
    """处理单个视频的包装函数，用于多进程"""
    video_path, detector = args
    try:
        no_change, info = detector.detect_no_change(video_path)
        return video_path, no_change, info
    except Exception as e:
        return video_path, False, f"处理出错: {str(e)}"


def process_videos_batch(video_paths: List[str], 
                        detector: VideoChangeDetector,
                        num_processes: Optional[int] = None) -> List[Tuple[str, bool, str]]:
    """
    批量处理视频
    
    Args:
        video_paths: 视频文件路径列表
        detector: 视频变化检测器
        num_processes: 进程数，默认为CPU核心数
        
    Returns:
        结果列表: [(视频路径, 是否无变化, 详细信息), ...]
    """
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), len(video_paths))
    
    # 准备参数
    args_list = [(path, detector) for path in video_paths]
    
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_single_video, args_list))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="检测视频前4秒是否完全没有变化")
    parser.add_argument("input", help="输入视频文件路径或包含视频的目录")
    parser.add_argument("-o", "--output", help="输出结果文件路径")
    parser.add_argument("-e", "--extensions", default="mp4,avi,mov,mkv,flv,wmv", 
                       help="视频文件扩展名，用逗号分隔")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                       help="并行处理的进程数（默认为CPU核心数）")
    parser.add_argument("--resize-width", type=int, default=160,
                       help="处理时的帧宽度（越小越快）")
    parser.add_argument("--resize-height", type=int, default=120,
                       help="处理时的帧高度（越小越快）")
    parser.add_argument("--ssim-threshold", type=float, default=0.98,
                       help="SSIM相似度阈值")
    parser.add_argument("--hist-threshold", type=float, default=0.99,
                       help="直方图相似度阈值")
    parser.add_argument("--hash-threshold", type=int, default=5,
                       help="感知哈希差异阈值")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="显示详细输出")
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = VideoChangeDetector(
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        ssim_threshold=args.ssim_threshold,
        hist_threshold=args.hist_threshold,
        hash_threshold=args.hash_threshold
    )
    
    # 收集视频文件
    video_paths = []
    extensions = [ext.strip().lower() for ext in args.extensions.split(',')]
    
    if os.path.isfile(args.input):
        # 单个文件
        video_paths = [args.input]
    elif os.path.isdir(args.input):
        # 目录中的所有视频文件
        for ext in extensions:
            pattern = f"*.{ext}"
            video_paths.extend(Path(args.input).glob(pattern))
            video_paths.extend(Path(args.input).glob(pattern.upper()))
        video_paths = [str(p) for p in video_paths]
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    if not video_paths:
        print("未找到视频文件")
        sys.exit(1)
    
    print(f"找到 {len(video_paths)} 个视频文件")
    
    # 处理视频
    start_time = time.time()
    results = process_videos_batch(video_paths, detector, args.jobs)
    end_time = time.time()
    
    # 统计结果
    no_change_count = sum(1 for _, no_change, _ in results if no_change)
    total_count = len(results)
    
    print(f"\n处理完成!")
    print(f"总数: {total_count}")
    print(f"无变化: {no_change_count}")
    print(f"有变化: {total_count - no_change_count}")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    print(f"平均速度: {total_count / (end_time - start_time):.2f} 个/秒")
    
    # 输出详细结果
    if args.verbose:
        print("\n详细结果:")
        for video_path, no_change, info in results:
            status = "无变化" if no_change else "有变化"
            print(f"{status}: {video_path} - {info}")
    
    # 保存结果到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("视频路径,是否无变化,详细信息\n")
            for video_path, no_change, info in results:
                f.write(f'"{video_path}",{no_change},"{info}"\n')
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()