"""
精细LOD构建脚本
专门用于处理大规模点云的精细LOD构建，避免层级间点数差异过大

作者：基于原始LOD脚本改进
"""

from multiprocessing import Pool, cpu_count, Manager
import os
import time
import struct
from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np
from scipy.spatial import KDTree

from tqdm import tqdm
from common import read_gaussian_file, write_gaussian_file
from tile import TileId
from point import Point

point_num_per_update = 1000

def build_fine_lod_tiles_for_parent(parent_tile_id: TileId, children_tile_ids: List[TileId], 
                                   input_dir: str, output_dir: str, distance_threshold: float, 
                                   target_reduction_ratio: float, progress_queue):
    """
    处理单个父级瓦片的精细LOD构建
    
    Args:
        target_reduction_ratio: 目标减少比例，例如0.5表示减少到50%
    """
    try:
        # 检测输入文件格式
        file_format = "splat"  # 默认格式
        for child_tile_id in children_tile_ids:
            # 检查是否存在PLY格式的文件
            ply_path = child_tile_id.getFilePath(input_dir, ".ply")
            if os.path.exists(ply_path):
                file_format = "ply"
                break

        ext = ".ply" if file_format == "ply" else ".splat"
        parent_tile_file_path = parent_tile_id.getFilePath(output_dir, ext)

        parent_points = []
        for child_tile_id in children_tile_ids:
            child_tile_file_path = child_tile_id.getFilePath(input_dir, ext)
            if os.path.exists(child_tile_file_path):
                points = read_gaussian_file(child_tile_file_path)
                parent_points.extend(points)

        point_num = len(parent_points)
        if point_num == 0:
            return []

        # 计算目标点数
        target_point_num = max(1, int(point_num * target_reduction_ratio))
        
        # 如果点数已经很少，直接返回
        if point_num <= target_point_num * 1.2:  # 20%的容差
            lod_points = parent_points
        else:
            # 使用分层聚类策略
            lod_points = hierarchical_clustering_lod(parent_points, target_point_num, distance_threshold, progress_queue)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(parent_tile_file_path), exist_ok=True)
        
        # 写入文件
        if lod_points:
            write_gaussian_file(parent_tile_file_path, lod_points)

        # 通知主进程任务完成
        progress_queue.put(None)
        
        return len(lod_points)

    except Exception as e:
        print(f"处理瓦片 {parent_tile_id} 时出错: {e}")
        import traceback
        print("完整错误堆栈:")
        traceback.print_exc()
        progress_queue.put(None)
        return 0

def hierarchical_clustering_lod(points: List[Point], target_point_num: int, 
                               distance_threshold: float, progress_queue) -> List[Point]:
    """
    分层聚类LOD算法，更精细地控制点数减少
    """
    if len(points) <= target_point_num:
        return points
    
    # 提取所有点的位置
    positions = np.array([point.position for point in points])
    
    # 构建 KDTree
    kdtree = KDTree(positions)
    visited = np.zeros(len(points), dtype=bool)
    lod_points = []
    
    # 计算自适应距离阈值
    adaptive_threshold = distance_threshold
    
    # 如果预期聚类后点数仍然太多，增加距离阈值
    estimated_clusters = estimate_cluster_count(positions, adaptive_threshold)
    if estimated_clusters > target_point_num * 1.5:
        adaptive_threshold *= 1.5
    elif estimated_clusters < target_point_num * 0.7:
        adaptive_threshold *= 0.8
    
    last_progress_update = 0
    for i in range(len(points)):
        # 每隔1000个点或处理完成时通知主进程
        if i % point_num_per_update == 0 or i == len(points) - 1:
            # 计算当前进度百分比
            current_progress = i / len(points)
            # 发送增量进度更新
            progress_increment = current_progress - last_progress_update
            if progress_increment > 0:
                progress_queue.put(progress_increment)
                last_progress_update = current_progress

        if visited[i]:
            continue

        # 查询当前点的邻域
        indices = kdtree.query_ball_point(positions[i], adaptive_threshold)

        # 标记这些点为已访问
        visited[indices] = True

        # 提取聚类中的点
        cluster_points = [points[j] for j in indices]

        # 计算加权平均点
        merged_point = compute_weighted_average_point(cluster_points)
        lod_points.append(merged_point)

        # 如果已经达到目标点数，停止聚类
        if len(lod_points) >= target_point_num:
            # 确保发送最终的进度更新
            if last_progress_update < 1.0:
                progress_queue.put(1.0 - last_progress_update)
            break

    # 确保循环结束时发送完整的进度更新
    if last_progress_update < 1.0:
        progress_queue.put(1.0 - last_progress_update)
    
    # 如果点数仍然太多，进行二次采样
    if len(lod_points) > target_point_num:
        lod_points = importance_sampling(lod_points, target_point_num)
    
    return lod_points

def estimate_cluster_count(positions: np.ndarray, distance_threshold: float) -> int:
    """
    估算给定距离阈值下的聚类数量
    """
    # 简单估算：使用网格方法
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    
    grid_size = distance_threshold
    grid_dims = np.ceil((max_pos - min_pos) / grid_size).astype(int)
    
    # 估算非空网格数量
    estimated_clusters = min(len(positions), np.prod(grid_dims))
    return int(estimated_clusters * 0.7)  # 考虑重叠，乘以0.7

def compute_weighted_average_point(cluster_points: List[Point]) -> Point:
    """
    计算聚类点的加权平均
    """
    if len(cluster_points) == 1:
        return cluster_points[0]
    
    # 计算权重，基于透明度
    raw_weights = np.array([point.color[3] / 255.0 for point in cluster_points])
    
    # 如果所有权重都为0或接近0，使用均等权重
    if np.sum(raw_weights) < 1e-6:
        weights = np.ones(len(cluster_points)) / len(cluster_points)
    else:
        weights = np.maximum(raw_weights, 1e-6)
        weights = weights / np.sum(weights)
    
    # 计算加权平均位置
    weighted_positions = np.average([point.position for point in cluster_points], axis=0, weights=weights)
    
    # 计算加权平均颜色
    weighted_color = np.average([point.color for point in cluster_points], axis=0, weights=weights)
    
    # 计算加权平均旋转
    weighted_rotation = np.average([point.rotation for point in cluster_points], axis=0, weights=weights)
    
    # 计算点的分布范围作为新的缩放
    cluster_positions = np.array([point.position for point in cluster_points])
    min_pos = max_pos = np.array(weighted_positions)
    
    for point in cluster_points:
        p1 = np.array(point.position) - np.array(point.scale)
        p2 = np.array(point.position) + np.array(point.scale)
        min_pos = np.minimum(min_pos, p1)
        max_pos = np.maximum(max_pos, p2)
    
    weighted_scale = (max_pos - min_pos) / 2
    
    # 确保数值在有效范围内
    weighted_color = np.clip(weighted_color, 0, 255).astype(int)
    weighted_rotation = np.clip(weighted_rotation, 0, 255).astype(int)
    
    return Point(tuple(weighted_positions), tuple(weighted_color), tuple(weighted_scale), tuple(weighted_rotation))

def importance_sampling(points: List[Point], target_count: int) -> List[Point]:
    """
    基于重要性的采样，保留最重要的点
    """
    if len(points) <= target_count:
        return points
    
    # 计算每个点的重要性分数（基于透明度和尺寸）
    importance_scores = []
    for point in points:
        alpha = point.color[3] / 255.0
        scale_volume = np.prod(point.scale)
        importance = alpha * scale_volume
        importance_scores.append(importance)
    
    # 按重要性排序并选择前target_count个点
    sorted_indices = np.argsort(importance_scores)[::-1]  # 降序
    selected_indices = sorted_indices[:target_count]
    
    return [points[i] for i in selected_indices]

def main_build_fine_lod_tiles(input_dir: str, output_dir: str,
                             enu_origin: Tuple[float, float] = (0.0, 0.0),
                             tile_zoom: int = 20, tile_resolution: float = 0.05,
                             target_reduction_ratio: float = 0.6):
    """
    构建精细LOD瓦片
    
    Args:
        target_reduction_ratio: 目标减少比例，0.6表示减少到60%
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 计算距离阈值
    level_diff = 20 - tile_zoom
    if level_diff == 0:
        distance_threshold = tile_resolution
    else:
        # 使用更保守的增长因子
        distance_threshold = tile_resolution * (1.2 ** level_diff)
    
    print(f"精细LOD级别 {tile_zoom}: 距离阈值 = {distance_threshold:.4f}米, 目标减少比例 = {target_reduction_ratio}")

    # 读取所有高斯文件
    gaussian_files = [f for f in os.listdir(input_dir) if f.endswith('.splat') or f.endswith('.ply')]
    
    # 从文件中解析出所有的瓦片
    gaussian_tiles: List[TileId] = []
    for gaussian_file in gaussian_files:
        tile_id = TileId.fromString(gaussian_file)
        gaussian_tiles.append(tile_id)

    parent_tiles = defaultdict(list)
    for tile_id in gaussian_tiles:
        parent_tile_id = tile_id.getParent()
        parent_tiles[parent_tile_id].append(tile_id)

    # 初始化进度队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化进度条 - 使用浮点数以支持增量更新
    total_tasks = len(parent_tiles)
    pbar = tqdm(total=float(total_tasks), desc="Building fine LOD", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理每个父级瓦片
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for parent_tile_id, children_tile_ids in parent_tiles.items():
            tasks.append(pool.apply_async(build_fine_lod_tiles_for_parent,
                                        (parent_tile_id, children_tile_ids, input_dir, output_dir,
                                         distance_threshold, target_reduction_ratio, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        total_output_points = 0
        task_results = {}  # 存储任务结果

        while completed_tasks < total_tasks:
            progress_update = progress_queue.get()

            if progress_update is None:
                completed_tasks += 1
                # 每完成一个任务，进度条增加1
                pbar.update(1.0)
            else:
                # 处理子任务内部的进度更新（如果需要的话）
                # 这里暂时不处理，因为我们主要关注任务完成度
                pass

        # 收集所有任务的结果
        for i, task in enumerate(tasks):
            try:
                result = task.get(timeout=1)  # 设置超时避免死锁
                if isinstance(result, int):
                    total_output_points += result
                task_results[i] = result
            except Exception as e:
                print(f"任务 {i} 执行出错: {e}")
                task_results[i] = 0

    pbar.close()
    print(f"精细LOD构建完成，输出点数: {total_output_points:,}")
    print(f"成功处理 {len([r for r in task_results.values() if r > 0])} / {total_tasks} 个瓦片")

if __name__ == "__main__":
    # 示例用法
    input_dir = "path/to/input"
    output_dir = "path/to/output"
    main_build_fine_lod_tiles(input_dir, output_dir, tile_zoom=19, target_reduction_ratio=0.7)
