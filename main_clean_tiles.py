import os
from multiprocessing import Pool, cpu_count, Manager
import struct
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from common import get_point_num, getPointSize, read_splat_file, write_splat_file, read_gaussian_file, write_gaussian_file

from point import Point
from tile import TileId

point_num_per_update = 1000

def write_splat_to_file(tile_file_path: str, point_size: int, min_alpha: float, max_scale: float, write_file, file_num: int, progress_queue):

    point_i = 0
    point_update = 0
    point_num = get_point_num(tile_file_path)

    with open(tile_file_path, 'rb') as f:
        while point_i < point_num:

            point_data = f.read(point_size)  # 3个 Float32，每个4字节
            if not point_data:
                break

            # 每隔1000个点通知主进程一次
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num / file_num
                progress_queue.put(progress_update)
                point_update = point_i

            point_i += 1

            position = struct.unpack('3f', point_data[0:12])
            scale = struct.unpack('3f', point_data[12:24])
            color = struct.unpack('4B', point_data[24:28])
            rotation = struct.unpack('4B', point_data[28:32])

            if color[3] < min_alpha or max(abs(s) for s in scale) > max_scale:
                continue  # 跳过无效点

            write_file.write(point_data)



# 清理单个瓦片
def clean_tile(tile_id: TileId, tile_files: List[str], output_dir: str,
               min_alpha: float, max_scale: float, flyers_num: float , flyers_dis: float,
               progress_queue):
    """
    清理单个瓦片
    """
    import time
    start_time = time.time()

    try:
        # 检测输入文件格式
        file_format = "ply" if any(f.endswith('.ply') for f in tile_files) else "splat"
        ext = ".ply" if file_format == "ply" else ".splat"
        output_tile_file_path = tile_id.getFilePath(output_dir, ext)
        if os.path.exists(output_tile_file_path):
            # print(f"Tile {tile_id.toString()} already exists, skipping.")
            progress_queue.put(1)
            progress_queue.put(None)  # 使用 None 作为任务完成的信号
            return

        if flyers_num > 0:
            all_points = []
            point_i = 0
            point_update = 0

            # 添加文件读取进度反馈
            print(f"瓦片 {tile_id.toString()}: 正在读取 {len(tile_files)} 个文件")
            for file_idx, tile_file_path in enumerate(tile_files):
                file_name = os.path.basename(tile_file_path)
                print(f"  [{file_idx + 1}/{len(tile_files)}] 读取: {file_name}")
                try:
                    points = read_gaussian_file(tile_file_path)  # 支持 .splat 和 .ply 格式
                    all_points.extend(points)
                    print(f"    ✓ 读取 {len(points):,} 个点，总计 {len(all_points):,} 个点")
                except Exception as e:
                    print(f"    ✗ 读取失败: {e}")
                    continue

                # 发送文件读取进度
                file_progress = (file_idx + 1) / len(tile_files) * 0.3  # 文件读取占30%进度
                progress_queue.put(file_progress)

            all_point_num = len(all_points)
            print(f"总共读取了 {all_point_num} 个点")

            # 检查点数量，如果太大则警告
            if all_point_num > 1000000:  # 超过100万个点
                print(f"⚠️  警告: 点数量很大 ({all_point_num:,})，处理可能需要较长时间")
                print("   建议: 可以考虑调整参数或分批处理")
            elif all_point_num > 500000:  # 超过50万个点
                print(f"ℹ️  信息: 点数量较多 ({all_point_num:,})，正在处理中...")

            mask = np.ones(all_point_num, dtype=bool)  # 初始化掩码，所有点都保留

            if all_point_num > 10:
                print(f"开始构建KDTree，点数: {all_point_num}")
                # 提取所有点的位置
                positions = np.array([point.position for point in all_points])

                # 发送KDTree构建开始进度
                progress_queue.put(0.4)  # 40%进度

                kdtree = KDTree(positions)
                print("KDTree构建完成")

                # 发送KDTree构建完成进度
                progress_queue.put(0.5)  # 50%进度

                # 移除飞点的逻辑
                k = max(3, min(int(flyers_num), all_point_num // 100))
                print(f"开始计算距离，k={k}")

                # 计算每个点的平均距离
                distances, _ = kdtree.query(positions, k=k+1)  # k+1 包括自身
                avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身，计算平均距离

                # 发送距离计算完成进度
                progress_queue.put(0.6)  # 60%进度

                # 计算阈值
                threshold = np.mean(avg_distances) + flyers_dis * np.std(avg_distances)
                print(f"飞点移除阈值: {threshold:.4f}")

                # 创建掩码，标记哪些点保留
                mask = avg_distances < threshold

                # 发送飞点检测完成进度
                progress_queue.put(0.7)  # 70%进度
            else:
                print(f"点数太少({all_point_num})，跳过飞点检测")

            # 过滤无效点
            print("开始过滤无效点...")
            invalid_count = 0
            for i in range(all_point_num):
                point = all_points[i]
                if point.color[3] < min_alpha or max(abs(s) for s in point.scale) > max_scale:
                    mask[i] = False  # 标记无效点
                    invalid_count += 1

                point_i += 1

                # 每隔1000个点通知主进程一次
                if point_i % point_num_per_update == 0 or point_i == all_point_num - 1:
                    progress_update = 0.7 + (point_i - point_update) / all_point_num * 0.2  # 70%-90%
                    progress_queue.put(progress_update)
                    point_update = point_i

            print(f"过滤完成，移除了 {invalid_count} 个无效点")

            # 发送过滤完成进度
            progress_queue.put(0.9)  # 90%进度

            # 应用掩码
            result_points = [all_points[i] for i in range(all_point_num) if mask[i]]
            print(f"最终保留 {len(result_points)} 个点")

            if len(result_points) > 0:
                print(f"写入文件: {output_tile_file_path}")
                write_gaussian_file(output_tile_file_path, result_points)
                print("文件写入完成")
            else:
                print("没有有效点，跳过文件写入")
        else:
            point_size = getPointSize()
            file_num = len(tile_files)

            write_file = open(output_tile_file_path, "w+b")
            writable = write_file.writable()
            if not writable:
                print(f"Error: Cannot write to {output_tile_file_path}.")
                progress_queue.put(1)
                progress_queue.put(None)
                return
            for tile_file_path in tile_files:
                write_splat_to_file(tile_file_path, point_size, min_alpha, max_scale, write_file, file_num, progress_queue)

        # 通知主进程任务完成
        progress_queue.put(1.0)  # 100%进度
        progress_queue.put(None)  # 使用 None 作为任务完成的信号

        elapsed_time = time.time() - start_time
        print(f"瓦片 {tile_id.toString()} 处理完成，耗时: {elapsed_time:.2f}秒")

    except Exception as e:
        print(f"处理瓦片 {tile_id.toString()} 时发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 即使出错也要通知主进程任务完成，避免进度条卡住
        progress_queue.put(1.0)
        progress_queue.put(None)


def main_clean_tiles(input_dir: str, output_dir: str,
                     min_alpha: float, max_scale: float, flyers_num: float , flyers_dis: float):

    """
    清理瓦片，使用多进程并行处理
    """

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有高斯文件 (支持 .splat 和 .ply 格式)
    gaussian_files = [f for f in os.listdir(input_dir) if f.endswith('.splat') or f.endswith('.ply')]

    # 从文件中解析出所有的瓦片
    gaussian_tiles = defaultdict(list)
    for gaussian_file in gaussian_files:
        tile_id = TileId.fromString(gaussian_file)
        gaussian_tiles[tile_id].append(os.path.join(input_dir, gaussian_file))

    # 初始化任务队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化进度条
    total_tasks = len(gaussian_tiles)
    pbar = tqdm(total=total_tasks, desc="Cleaning tiles", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理每个父级瓦片
    with Pool(processes=cpu_count() - 1) as pool:
        tasks = []
        for tile_id, tile_files in gaussian_tiles.items():
            tasks.append(pool.apply_async(clean_tile, (tile_id, tile_files, output_dir, min_alpha, max_scale, flyers_num, flyers_dis, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        while completed_tasks < total_tasks:
            progress_update = progress_queue.get()  # 等待子进程通知进度

            if progress_update is None:
                completed_tasks += 1  # 任务完成信号
            else:
                pbar.update(progress_update)  # 更新进度条

        # 等待所有任务完成
        for task in tasks:
            task.get()

    # 关闭进度条
    pbar.close()