#!/usr/bin/env python3
"""
优化版本的瓦片清理脚本
专门处理大PLY文件的性能问题
"""

import os
import time
import gc
from multiprocessing import Pool, cpu_count, Manager
from typing import Dict, List
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from common import get_point_num, read_gaussian_file, write_gaussian_file
from tile import TileId

point_num_per_update = 1000

def clean_tile_optimized(tile_id: TileId, tile_files: List[str], output_dir: str,
                        min_alpha: float, max_scale: float, flyers_num: float, flyers_dis: float,
                        progress_queue):
    """
    优化版本的瓦片清理函数
    """
    start_time = time.time()
    
    try:
        # 检测输入文件格式
        file_format = "ply" if any(f.endswith('.ply') for f in tile_files) else "splat"
        ext = ".ply" if file_format == "ply" else ".splat"
        output_tile_file_path = tile_id.getFilePath(output_dir, ext)
        
        if os.path.exists(output_tile_file_path):
            progress_queue.put(1)
            progress_queue.put(None)
            return

        print(f"处理瓦片: {tile_id.toString()}, 文件数: {len(tile_files)}")
        
        # 预估总点数
        total_estimated_points = 0
        file_info = []
        
        for tile_file_path in tile_files:
            try:
                point_count = get_point_num(tile_file_path)
                file_size = os.path.getsize(tile_file_path)
                file_info.append((tile_file_path, point_count, file_size))
                total_estimated_points += point_count
            except Exception as e:
                print(f"  跳过文件 {os.path.basename(tile_file_path)}: {e}")
                continue
        
        print(f"  预估总点数: {total_estimated_points:,}")
        
        # 根据点数选择处理策略
        if total_estimated_points > 500000:  # 大于50万个点
            print("  使用分批处理策略")
            result = process_large_tile(file_info, output_tile_file_path, min_alpha, max_scale, 
                                      flyers_num, flyers_dis, progress_queue)
        else:
            print("  使用标准处理策略")
            result = process_standard_tile(file_info, output_tile_file_path, min_alpha, max_scale,
                                         flyers_num, flyers_dis, progress_queue)
        
        elapsed_time = time.time() - start_time
        print(f"  瓦片处理完成，耗时: {elapsed_time:.2f}秒")
        
        progress_queue.put(1.0)
        progress_queue.put(None)
        
    except Exception as e:
        print(f"处理瓦片 {tile_id.toString()} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        progress_queue.put(1.0)
        progress_queue.put(None)

def process_standard_tile(file_info: List, output_path: str, min_alpha: float, max_scale: float,
                         flyers_num: float, flyers_dis: float, progress_queue):
    """
    标准处理策略 - 一次性加载所有点
    """
    all_points = []
    
    # 读取所有文件
    for i, (file_path, point_count, file_size) in enumerate(file_info):
        print(f"    读取文件 {i+1}/{len(file_info)}: {os.path.basename(file_path)} ({point_count:,} 点)")
        try:
            points = read_gaussian_file(file_path)
            all_points.extend(points)
            
            # 更新进度
            progress = (i + 1) / len(file_info) * 0.3
            progress_queue.put(progress)
            
        except Exception as e:
            print(f"      读取失败: {e}")
            continue
    
    if not all_points:
        print("    没有有效点")
        return False
    
    print(f"    总共读取: {len(all_points):,} 个点")
    
    # 应用过滤
    result_points = apply_filters(all_points, min_alpha, max_scale, flyers_num, flyers_dis, progress_queue)
    
    # 写入结果
    if result_points:
        print(f"    写入 {len(result_points):,} 个点到: {os.path.basename(output_path)}")
        write_gaussian_file(output_path, result_points)
        return True
    else:
        print("    没有点通过过滤")
        return False

def process_large_tile(file_info: List, output_path: str, min_alpha: float, max_scale: float,
                      flyers_num: float, flyers_dis: float, progress_queue):
    """
    大文件处理策略 - 分批处理以节省内存
    """
    print("    使用分批处理模式")
    
    # 按文件大小排序，先处理小文件
    file_info.sort(key=lambda x: x[2])
    
    batch_size = 100000  # 每批处理10万个点
    all_result_points = []
    
    current_batch = []
    current_batch_size = 0
    
    for i, (file_path, point_count, file_size) in enumerate(file_info):
        print(f"    处理文件 {i+1}/{len(file_info)}: {os.path.basename(file_path)}")
        
        try:
            points = read_gaussian_file(file_path)
            
            # 预过滤明显无效的点
            valid_points = []
            for point in points:
                if point.color[3] >= min_alpha and max(abs(s) for s in point.scale) <= max_scale:
                    valid_points.append(point)
            
            print(f"      预过滤后: {len(valid_points):,}/{len(points):,} 个点")
            
            current_batch.extend(valid_points)
            current_batch_size += len(valid_points)
            
            # 如果批次足够大或是最后一个文件，处理当前批次
            if current_batch_size >= batch_size or i == len(file_info) - 1:
                if current_batch:
                    print(f"      处理批次: {len(current_batch):,} 个点")
                    
                    # 对当前批次应用飞点移除
                    if flyers_num > 0 and len(current_batch) > 10:
                        batch_result = remove_flyers(current_batch, flyers_num, flyers_dis)
                    else:
                        batch_result = current_batch
                    
                    all_result_points.extend(batch_result)
                    print(f"      批次结果: {len(batch_result):,} 个点")
                
                # 清理内存
                current_batch = []
                current_batch_size = 0
                gc.collect()
            
            # 更新进度
            progress = (i + 1) / len(file_info) * 0.9
            progress_queue.put(progress)
            
        except Exception as e:
            print(f"      处理失败: {e}")
            continue
    
    # 写入最终结果
    if all_result_points:
        print(f"    最终结果: {len(all_result_points):,} 个点")
        write_gaussian_file(output_path, all_result_points)
        return True
    else:
        print("    没有点通过处理")
        return False

def apply_filters(points: List, min_alpha: float, max_scale: float, 
                 flyers_num: float, flyers_dis: float, progress_queue):
    """
    应用所有过滤器
    """
    print(f"    应用过滤器...")
    
    # 基本过滤
    valid_points = []
    for point in points:
        if point.color[3] >= min_alpha and max(abs(s) for s in point.scale) <= max_scale:
            valid_points.append(point)
    
    print(f"    基本过滤后: {len(valid_points):,}/{len(points):,} 个点")
    progress_queue.put(0.6)
    
    # 飞点移除
    if flyers_num > 0 and len(valid_points) > 10:
        result_points = remove_flyers(valid_points, flyers_num, flyers_dis)
        print(f"    飞点移除后: {len(result_points):,}/{len(valid_points):,} 个点")
    else:
        result_points = valid_points
    
    progress_queue.put(0.9)
    return result_points

def remove_flyers(points: List, flyers_num: float, flyers_dis: float):
    """
    移除飞点
    """
    if len(points) <= 10:
        return points
    
    print(f"      构建KDTree...")
    positions = np.array([point.position for point in points])
    kdtree = KDTree(positions)
    
    k = max(3, min(int(flyers_num), len(points) // 100))
    print(f"      计算距离 (k={k})...")
    
    distances, _ = kdtree.query(positions, k=k+1)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    
    threshold = np.mean(avg_distances) + flyers_dis * np.std(avg_distances)
    print(f"      飞点阈值: {threshold:.4f}")
    
    mask = avg_distances < threshold
    result_points = [points[i] for i in range(len(points)) if mask[i]]
    
    return result_points

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='优化版瓦片清理工具')
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--min_alpha', type=float, default=1.0, help='最小透明度')
    parser.add_argument('--max_scale', type=float, default=10000, help='最大缩放值')
    parser.add_argument('--flyers_num', type=float, default=25, help='飞点检测邻近点数')
    parser.add_argument('--flyers_dis', type=float, default=10, help='飞点检测距离因子')
    
    args = parser.parse_args()
    
    print("优化版瓦片清理工具")
    print("=" * 50)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"参数: min_alpha={args.min_alpha}, max_scale={args.max_scale}")
    print(f"飞点参数: flyers_num={args.flyers_num}, flyers_dis={args.flyers_dis}")
    
    # 确保输出目录存在
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 查找瓦片文件
    gaussian_files = [f for f in os.listdir(args.input) if f.endswith('.splat') or f.endswith('.ply')]
    gaussian_tiles = defaultdict(list)
    
    for gaussian_file in gaussian_files:
        tile_id = TileId.fromString(gaussian_file)
        gaussian_tiles[tile_id].append(os.path.join(args.input, gaussian_file))
    
    print(f"找到 {len(gaussian_tiles)} 个瓦片")
    
    # 使用多进程处理
    with Manager() as manager:
        progress_queue = manager.Queue()
        
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            tasks = []
            for tile_id, tile_files in gaussian_tiles.items():
                task = pool.apply_async(clean_tile_optimized, 
                                      (tile_id, tile_files, args.output, 
                                       args.min_alpha, args.max_scale, 
                                       args.flyers_num, args.flyers_dis, progress_queue))
                tasks.append(task)
            
            # 等待所有任务完成
            for task in tasks:
                task.get()
    
    print("所有瓦片处理完成!")

if __name__ == "__main__":
    main()
