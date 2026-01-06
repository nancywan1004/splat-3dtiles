import os
import time
import struct
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm

from common import getPointSize, read_gaussian_file, write_gaussian_file

from point import Point
from tile import TileId

import numpy as np

# 尝试导入GPU工具，如果不可用则使用scipy
try:
    from gpu_utils import GPUIndexWrapper, check_gpu_available, get_default_gpu_id
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUIndexWrapper = None

try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    KDTree = None


point_num_per_update = 1000



def build_lod_tiles_for_parent(parent_tile_id: TileId, children_tile_ids: List[TileId], input_dir: str, output_dir: str, distance_threshold: float, progress_queue, use_gpu: bool = False, gpu_id: int = 0):
    """
    处理单个父级瓦片的LOD构建
    """
    try:
        # 检测输入文件格式 - 改进的检测逻辑
        file_format = "ply"  # 默认为ply格式，因为现在大部分文件都是ply

        # 首先检查输入目录中实际存在的文件
        if os.path.exists(input_dir):
            existing_files = os.listdir(input_dir)
            # 检查是否有任何ply文件
            has_ply = any(f.endswith('.ply') for f in existing_files)
            has_splat = any(f.endswith('.splat') for f in existing_files)

            if has_ply:
                file_format = "ply"
            elif has_splat:
                file_format = "splat"
            else:
                # 如果目录为空或没有相关文件，尝试检查具体的子瓦片文件
                for child_tile_id in children_tile_ids:
                    # 检查是否存在PLY格式的文件
                    ply_path = child_tile_id.getFilePath(input_dir, ".ply")
                    splat_path = child_tile_id.getFilePath(input_dir, ".splat")
                    if os.path.exists(ply_path):
                        file_format = "ply"
                        break
                    elif os.path.exists(splat_path):
                        file_format = "splat"
                        break

        ext = ".ply" if file_format == "ply" else ".splat"
        parent_tile_file_path = parent_tile_id.getFilePath(output_dir, ext)

        parent_points = []
        for child_tile_id in children_tile_ids:
            child_tile_file_path = child_tile_id.getFilePath(input_dir, ext)
            if os.path.exists(child_tile_file_path):
                try:
                    points = read_gaussian_file(child_tile_file_path)  # 支持 .splat 和 .ply 格式
                    if points:
                        parent_points.extend(points)
                        print(f"成功读取 {len(points)} 个点从 {os.path.basename(child_tile_file_path)}")
                    else:
                        print(f"警告: 文件 {os.path.basename(child_tile_file_path)} 为空")
                except Exception as e:
                    print(f"读取文件 {child_tile_file_path} 时出错: {e}")
                    continue

        lod_points = []
        point_num = len(parent_points)
        print(f"瓦片 {parent_tile_id}: 开始处理 {point_num:,} 个点，距离阈值: {distance_threshold:.4f}")

        if point_num == 0:
            print(f"瓦片 {parent_tile_id}: 没有点数据，跳过")
            return lod_points

        # 检查点数据的有效性
        try:
            # 提取所有点的位置
            positions = np.array([point.position for point in parent_points])

            # 检查位置数据是否有效
            if not np.isfinite(positions).all():
                print(f"瓦片 {parent_tile_id}: 发现无效的位置数据，尝试清理...")
                # 过滤掉无效的点
                valid_indices = np.isfinite(positions).all(axis=1)
                valid_points = [parent_points[i] for i in range(len(parent_points)) if valid_indices[i]]
                if len(valid_points) == 0:
                    print(f"瓦片 {parent_tile_id}: 所有点都无效，跳过")
                    return lod_points
                parent_points = valid_points
                positions = np.array([point.position for point in parent_points])
                point_num = len(parent_points)
                print(f"瓦片 {parent_tile_id}: 清理后剩余 {point_num:,} 个有效点")

            # 构建 KDTree 或 GPU索引
            print(f"瓦片 {parent_tile_id}: 构建空间索引...")
            if use_gpu and GPU_AVAILABLE and GPUIndexWrapper is not None:
                try:
                    kdtree = GPUIndexWrapper(positions, use_gpu=True, gpu_id=gpu_id)
                except Exception as e:
                    print(f"瓦片 {parent_tile_id}: GPU索引构建失败，回退到CPU: {e}")
                    if SCIPY_AVAILABLE and KDTree is not None:
                        kdtree = KDTree(positions)
                    else:
                        raise RuntimeError("既没有GPU也没有scipy，无法构建索引")
            else:
                if SCIPY_AVAILABLE and KDTree is not None:
                    kdtree = KDTree(positions)
                else:
                    raise RuntimeError("scipy不可用，无法构建索引")
            
            visited = np.zeros(point_num, dtype=bool)
            print(f"瓦片 {parent_tile_id}: 空间索引构建完成")

        except Exception as e:
            print(f"瓦片 {parent_tile_id}: 构建KDTree时出错: {e}")
            raise

        last_progress_update = 0
        cluster_count = 0

        try:
            for i in range(point_num):
                # 每隔1000个点或处理完成时通知主进程
                if i % point_num_per_update == 0 or i == point_num - 1:
                    # 计算当前进度百分比
                    current_progress = i / point_num
                    # 发送增量进度更新
                    progress_increment = current_progress - last_progress_update
                    if progress_increment > 0:
                        progress_queue.put(progress_increment)
                        last_progress_update = current_progress

                if visited[i]:
                    continue

                try:
                    # 查询当前点的邻域
                    indices = kdtree.query_ball_point(positions[i], distance_threshold)

                    if not indices:  # 如果没有找到邻域点
                        continue

                    # 标记这些点为已访问
                    visited[indices] = True

                    # 提取聚类中的点
                    cluster_points = [parent_points[j] for j in indices]
                    cluster_count += 1

                    # 计算权重，基于透明度，但确保至少有最小权重
                    raw_weights = np.array([point.color[3] / 255.0 for point in cluster_points])

                    # 如果所有权重都为0或接近0，使用均等权重
                    if np.sum(raw_weights) < 1e-6:
                        weights = np.ones(len(cluster_points)) / len(cluster_points)
                    else:
                        # 确保权重至少有一个最小值，避免数值问题
                        weights = np.maximum(raw_weights, 1e-6)
                        weights = weights / np.sum(weights)  # 归一化

                    # 验证权重
                    if not np.isfinite(weights).all() or np.sum(weights) < 1e-6:
                        weights = np.ones(len(cluster_points)) / len(cluster_points)

                    try:
                        # 计算加权平均位置
                        weighted_positions = np.average([point.position for point in cluster_points], axis=0, weights=weights)
                        # 计算加权平均颜色
                        weighted_color = np.average([point.color for point in cluster_points], axis=0, weights=weights)
                        # 计算加权平均缩放
                        # weighted_scale = np.average([point.scale for point in cluster_points], axis=0, weights=weights)
                        # 计算加权平均旋转
                        weighted_rotation = np.average([point.rotation for point in cluster_points], axis=0, weights=weights)
                    except Exception as e:
                        # 如果加权平均仍然失败，使用简单平均作为备选方案
                        print(f"瓦片 {parent_tile_id}: 加权平均计算失败，使用简单平均: {e}")
                        weighted_positions = np.mean([point.position for point in cluster_points], axis=0)
                        weighted_color = np.mean([point.color for point in cluster_points], axis=0)
                        weighted_rotation = np.mean([point.rotation for point in cluster_points], axis=0)

                    # 计算点的分布范围
                    cluster_positions = np.array([point.position for point in cluster_points])

                    min_pos = max_pos = np.array(weighted_positions)

                    # 计算每个点的边界
                    for point in cluster_points:
                        p1 = np.array(point.position) - np.array(point.scale)
                        p2 = np.array(point.position) + np.array(point.scale)
                        min_pos = np.minimum(min_pos, p1)
                        max_pos = np.maximum(max_pos, p2)

                    weighted_scale = (max_pos - min_pos) / 2

                    weighted_color = np.clip(weighted_color, 0, 255)  # 限制范围
                    weighted_color = np.round(weighted_color).astype(int)  # 取整并转换为整数

                    weighted_rotation = np.clip(weighted_rotation, 0, 255)  # 限制范围
                    weighted_rotation = np.round(weighted_rotation).astype(int)  # 取整并转换为整数

                    # 处理球谐系数 - 如果原始点有球谐系数，计算加权平均
                    weighted_sh_coeffs = None
                    if cluster_points and hasattr(cluster_points[0], 'sh_coeffs') and cluster_points[0].sh_coeffs:
                        try:
                            # 收集所有点的球谐系数
                            sh_coeffs_list = []
                            for point in cluster_points:
                                if hasattr(point, 'sh_coeffs') and point.sh_coeffs:
                                    sh_coeffs_list.append(point.sh_coeffs)
                                else:
                                    # 如果某个点没有球谐系数，用零填充
                                    sh_coeffs_list.append([0.0] * 48)

                            if sh_coeffs_list:
                                # 计算加权平均球谐系数
                                weighted_sh_coeffs = np.average(sh_coeffs_list, axis=0, weights=weights).tolist()
                        except Exception as e:
                            weighted_sh_coeffs = None

                    lod_points.append(Point(weighted_positions, weighted_color, weighted_scale, weighted_rotation, weighted_sh_coeffs))

                except Exception as e:
                    print(f"瓦片 {parent_tile_id}: 处理点 {i} 时出错: {e}")
                    continue

            # 确保循环结束时发送完整的进度更新
            if last_progress_update < 1.0:
                progress_queue.put(1.0 - last_progress_update)

            print(f"瓦片 {parent_tile_id}: 聚类完成，生成 {cluster_count} 个聚类，输出 {len(lod_points)} 个LOD点")

        except Exception as e:
            print(f"瓦片 {parent_tile_id}: 聚类过程出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理GPU资源
            if 'kdtree' in locals() and isinstance(kdtree, GPUIndexWrapper):
                try:
                    kdtree.cleanup()
                except Exception:
                    pass

        # 确保输出目录存在
        os.makedirs(os.path.dirname(parent_tile_file_path), exist_ok=True)

        # 写入文件
        if lod_points:
            write_gaussian_file(parent_tile_file_path, lod_points)
            print(f"瓦片 {parent_tile_id}: 成功写入 {len(lod_points)} 个点到 {os.path.basename(parent_tile_file_path)}")
        else:
            print(f"瓦片 {parent_tile_id}: 没有生成LOD点，跳过写入")

        # 通知主进程任务完成
        progress_queue.put(None)  # 使用 None 作为任务完成的信号
        return len(lod_points)  # 返回生成的LOD点数，用于统计
    except Exception as e:
        print(f"Error in build_lod_tiles_for_parent (瓦片 {parent_tile_id}): {e}")
        import traceback
        print("完整错误堆栈:")
        traceback.print_exc()
        progress_queue.put(None)  # 确保主进程不会阻塞
        return 0  # 返回0表示失败


def main_build_lod_tiles(input_dir: str, output_dir: str,
                         enu_origin: Tuple[float, float] = (0.0, 0.0),
                         tile_zoom: int = 20, tile_resolution: float = 0.05,
                         lod_factor: float = 1.5, use_gpu: bool = False, gpu_id: int = 0):
    """
    构建LOD瓦片，使用多进程并行处理

    Args:
        lod_factor: LOD因子，控制聚类的激进程度。值越小越精细，默认1.5
                   建议范围：1.2-2.0，1.2最精细，2.0最激进
    """


    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用更精细的距离阈值计算
    # 原公式：distance_threshold = tile_resolution * (2** (20 - tile_zoom))
    # 新公式：使用更小的增长因子，保留更多细节
    level_diff = 20 - tile_zoom
    if level_diff == 0:
        distance_threshold = tile_resolution
    else:
        # 使用可调节的LOD因子，而不是固定的2倍增长
        distance_threshold = tile_resolution * (lod_factor ** level_diff)

    # 自适应调整：对于高密度场景，使用更小的阈值
    gaussian_files = [f for f in os.listdir(input_dir) if f.endswith('.splat') or f.endswith('.ply')]
    if len(gaussian_files) > 1000:  # 高密度场景
        distance_threshold *= 0.7  # 减小30%，保留更多细节
        print(f"检测到高密度场景({len(gaussian_files)}个文件)，调整距离阈值")
    elif len(gaussian_files) > 500:  # 中等密度场景
        distance_threshold *= 0.85  # 减小15%
        print(f"检测到中等密度场景({len(gaussian_files)}个文件)，轻微调整距离阈值")

    print(f"LOD级别 {tile_zoom}: 距离阈值 = {distance_threshold:.4f}米 (LOD因子: {lod_factor}, 文件数: {len(gaussian_files)})")

    # 从文件中解析出所有的瓦片
    gaussian_tiles: List[TileId] = []
    for gaussian_file in gaussian_files:
        tile_id = TileId.fromString(gaussian_file)
        gaussian_tiles.append(tile_id)

    parent_tiles = defaultdict(list)
    for tile_id in gaussian_tiles:
        # 计算跳级后的父瓦片ID：直接使用目标层级，而不是getParent()
        # 这样可以确保生成的瓦片文件名与目标层级一致
        level_diff = tile_id.z - tile_zoom  # 计算层级差
        parent_x = tile_id.x >> level_diff  # 右移level_diff位，相当于除以2^level_diff
        parent_y = tile_id.y >> level_diff
        parent_tile_id = TileId(parent_x, parent_y, tile_zoom)
        parent_tiles[parent_tile_id].append(tile_id)

    # 初始化进度队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 检查GPU可用性
    if use_gpu:
        if GPU_AVAILABLE:
            gpu_available, gpu_count = check_gpu_available()
            if not gpu_available:
                print("⚠️  请求使用GPU，但GPU不可用，回退到CPU")
                use_gpu = False
            else:
                print(f"✓ 检测到 {gpu_count} 个GPU，使用GPU ID: {gpu_id}")
        else:
            print("⚠️  请求使用GPU，但GPU工具不可用，回退到CPU")
            use_gpu = False
    
    # 初始化进度条 - 使用浮点数以支持增量更新
    total_tasks = len(parent_tiles)
    pbar = tqdm(total=float(total_tasks), desc="Building lod", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理每个父级瓦片
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for parent_tile_id, children_tile_ids in parent_tiles.items():
            tasks.append(pool.apply_async(build_lod_tiles_for_parent, (parent_tile_id, children_tile_ids, input_dir, output_dir, distance_threshold, progress_queue, use_gpu, gpu_id)))

        # 等待所有任务完成
        completed_tasks = 0
        task_results = {}  # 存储任务结果

        while completed_tasks < total_tasks:
            progress_update = progress_queue.get()  # 等待子进程通知进度

            if progress_update is None:
                completed_tasks += 1  # 任务完成信号
                pbar.update(1.0)  # 每完成一个任务，进度条增加1
            else:
                # 处理子任务内部的进度更新（如果需要的话）
                pass

        # 收集所有任务的结果
        for i, task in enumerate(tasks):
            try:
                result = task.get(timeout=1)  # 设置超时避免死锁
                task_results[i] = result
            except Exception as e:
                print(f"任务 {i} 执行出错: {e}")
                task_results[i] = None

    # 关闭进度条
    pbar.close()
    successful_tasks = [r for r in task_results.values() if r is not None and r > 0]
    print(f"成功处理 {len(successful_tasks)} / {total_tasks} 个瓦片")