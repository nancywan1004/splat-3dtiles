#!/usr/bin/env python3
"""
Splat文件切片工具
基于3D空间坐标将splat文件切分为多个小的splat文件
不依赖地理坐标系统，使用简单的3D网格切片

作者：基于现有splat-3dtiles代码修改
"""

import argparse
import json
import os
import struct
import time
from multiprocessing import Pool, cpu_count, Manager
from typing import Dict, Tuple, List
from collections import defaultdict

from tqdm import tqdm
from common import get_point_num, getPointSize, read_splat_file, write_splat_file
from point import Point, compute_box


class SimpleTileId:
    """简化的瓦片ID，基于3D网格坐标"""
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def toString(self) -> str:
        """将瓦片 ID 转换为字符串"""
        return f"tile_{self.x}_{self.y}_{self.z}"

    def getFilePath(self, output_dir: str, ext: str = ".splat") -> str:
        """获取瓦片文件路径"""
        splat_file = f"{self.toString()}{ext}"
        output_file = os.path.join(output_dir, splat_file)
        return output_file

    def __eq__(self, other):
        if not isinstance(other, SimpleTileId):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))


class SimpleTileManager:
    """简化的瓦片管理器，基于3D空间网格切片"""

    def __init__(self, grid_size: float = 10.0, bounds: Tuple[float, float, float, float, float, float] = None):
        """
        初始化简单瓦片管理器

        Args:
            grid_size: 网格大小（每个瓦片的边长）
            bounds: 空间边界 (min_x, min_y, min_z, max_x, max_y, max_z)，如果为None则自动计算
        """
        self.grid_size = grid_size
        self.bounds = bounds
        self.auto_bounds = bounds is None

        # 如果需要自动计算边界，这些值会在第一次处理时设置
        if self.auto_bounds:
            self.min_x = float('inf')
            self.min_y = float('inf')
            self.min_z = float('inf')
            self.max_x = float('-inf')
            self.max_y = float('-inf')
            self.max_z = float('-inf')
        else:
            self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z = bounds

    def update_bounds(self, position: Tuple[float, float, float]):
        """更新边界（仅在自动计算边界时使用）"""
        if not self.auto_bounds:
            return

        x, y, z = position
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.min_z = min(self.min_z, z)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        self.max_z = max(self.max_z, z)

    def getTileId(self, position: Tuple[float, float, float]) -> SimpleTileId:
        """
        根据3D位置获取瓦片ID

        Args:
            position: 3D坐标 (x, y, z)

        Returns:
            SimpleTileId: 瓦片ID
        """
        x, y, z = position

        # 如果是自动边界模式，先更新边界
        if self.auto_bounds:
            self.update_bounds(position)

        # 计算网格坐标
        tile_x = int((x - self.min_x) // self.grid_size)
        tile_y = int((y - self.min_y) // self.grid_size)
        tile_z = int((z - self.min_z) // self.grid_size)

        return SimpleTileId(tile_x, tile_y, tile_z)

    def get_bounds_info(self) -> str:
        """获取边界信息字符串"""
        return f"Bounds: X[{self.min_x:.2f}, {self.max_x:.2f}] Y[{self.min_y:.2f}, {self.max_y:.2f}] Z[{self.min_z:.2f}, {self.max_z:.2f}]"


def write_point_to_tile_file(point_data: bytes, tile_id: SimpleTileId, tile_files: Dict, output_dir: str):
    """将单个高斯点的数据写入分块文件"""
    write_file = tile_files.get(tile_id)
    if not write_file:
        output_file = tile_id.getFilePath(output_dir)
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        write_file = open(output_file, "w+b")
        tile_files[tile_id] = write_file

    write_file.write(point_data)


def split_splat_file_by_count(input_file: str, output_dir: str, points_per_tile: int, progress_queue, shared_tile_info) -> None:
    """
    按固定点数将单个splat文件切分为多个瓦片文件，并计算每个瓦片的边界框信息

    Args:
        input_file: 输入splat文件路径
        output_dir: 输出目录
        points_per_tile: 每个瓦片的点数
        progress_queue: 进度队列
        shared_tile_info: 共享的瓦片信息字典
    """
    point_size = getPointSize()
    point_num = get_point_num(input_file)
    point_i = 0
    point_update = 0
    point_num_per_update = 1000

    current_tile = 0
    current_tile_points = 0
    current_tile_data = []
    current_tile_positions = []  # 存储当前瓦片的位置信息用于计算边界框

    # 获取输入文件名（不含扩展名）作为前缀
    input_filename = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file, 'rb') as f:
        while point_i < point_num:
            point_data = f.read(point_size)
            if not point_data:
                break

            # 解析位置数据
            position_data = point_data[0:12]
            position = struct.unpack('3f', position_data)

            # 添加到当前瓦片
            current_tile_data.append(point_data)
            current_tile_positions.append(position)
            current_tile_points += 1
            point_i += 1

            # 如果当前瓦片已满或者是最后一个点，保存瓦片
            if current_tile_points >= points_per_tile or point_i == point_num:
                # 生成瓦片文件名
                tile_filename = f"{input_filename}_tile_{current_tile:04d}.splat"
                tile_filepath = os.path.join(output_dir, tile_filename)

                # 写入瓦片文件
                with open(tile_filepath, 'wb') as tile_file:
                    for data in current_tile_data:
                        tile_file.write(data)

                # 计算边界框信息
                tile_info = calculate_tile_info(current_tile_positions, tile_filename, current_tile_points, point_i - current_tile_points)

                # 将瓦片信息添加到共享字典
                shared_tile_info[tile_filename] = tile_info

                # 重置当前瓦片
                current_tile += 1
                current_tile_points = 0
                current_tile_data = []
                current_tile_positions = []

            # 每隔1000个点通知主进程一次
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num
                progress_queue.put(progress_update)
                point_update = point_i

    # 通知主进程任务完成
    progress_queue.put(None)


def calculate_tile_info(positions: List[Tuple[float, float, float]], filename: str, count: int, offset: int) -> Dict:
    """
    计算瓦片的边界框和其他信息

    Args:
        positions: 瓦片中所有点的位置列表
        filename: 瓦片文件名
        count: 点数
        offset: 在原文件中的偏移量（点索引）

    Returns:
        包含瓦片信息的字典
    """
    import numpy as np

    if not positions:
        return {
            "filename": filename,
            "count": count,
            "offset": offset,
            "bounding_box": {
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0]
            },
            "center": [0.0, 0.0, 0.0],
            "size": [0.0, 0.0, 0.0]
        }

    # 转换为numpy数组进行计算
    positions_array = np.array(positions)

    # 计算边界框
    min_coords = np.min(positions_array, axis=0)
    max_coords = np.max(positions_array, axis=0)

    # 计算中心点和尺寸
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords

    return {
        "filename": filename,
        "count": count,
        "offset": offset,
        "bounding_box": {
            "min": min_coords.tolist(),
            "max": max_coords.tolist()
        },
        "center": center.tolist(),
        "size": size.tolist()
    }


def split_splat_file_by_grid(input_file: str, output_dir: str, tile_manager: SimpleTileManager, progress_queue) -> None:
    """
    按网格将单个splat文件切分为多个瓦片文件（原有方法）

    Args:
        input_file: 输入splat文件路径
        output_dir: 输出目录
        tile_manager: 瓦片管理器
        progress_queue: 进度队列
    """
    point_size = getPointSize()
    point_num = get_point_num(input_file)
    point_i = 0
    point_update = 0
    point_num_per_update = 1000

    tile_files = {}

    with open(input_file, 'rb') as f:
        while point_i < point_num:
            point_data = f.read(point_size)
            if not point_data:
                break

            # 解析位置数据
            position_data = point_data[0:12]
            position = struct.unpack('3f', position_data)

            # 获取瓦片ID并写入文件
            tile_id = tile_manager.getTileId(position)
            write_point_to_tile_file(point_data, tile_id, tile_files, output_dir)

            point_i += 1

            # 每隔1000个点通知主进程一次
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num
                progress_queue.put(progress_update)
                point_update = point_i

    # 关闭所有文件
    for tile_file in tile_files.values():
        tile_file.close()

    # 通知主进程任务完成
    progress_queue.put(None)


def calculate_bounds_from_files(input_files: List[str]) -> Tuple[float, float, float, float, float, float]:
    """
    从所有输入文件计算边界

    Args:
        input_files: 输入文件列表

    Returns:
        边界元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    """
    print("正在计算空间边界...")

    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    total_points = 0
    for input_file in input_files:
        points = read_splat_file(input_file)
        total_points += len(points)

        for point in points:
            x, y, z = point.position
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

    print(f"总点数: {total_points}")
    print(f"空间边界: X[{min_x:.2f}, {max_x:.2f}] Y[{min_y:.2f}, {max_y:.2f}] Z[{min_z:.2f}, {max_z:.2f}]")

    return min_x, min_y, min_z, max_x, max_y, max_z


def main_split_splat_files(input_path: str, output_dir: str,
                          mode: str = "count", points_per_tile: int = 4096,
                          grid_size: float = 10.0, auto_bounds: bool = True,
                          bounds: Tuple[float, float, float, float, float, float] = None):
    """
    主函数：将splat文件切分为多个瓦片文件

    Args:
        input_path: 输入路径（文件或目录）
        output_dir: 输出目录
        mode: 切片模式 ("count" 按点数切片, "grid" 按网格切片)
        points_per_tile: 每个瓦片的点数（仅在count模式下使用）
        grid_size: 网格大小（仅在grid模式下使用）
        auto_bounds: 是否自动计算边界（仅在grid模式下使用）
        bounds: 手动指定的边界（仅在grid模式下使用）
    """
    start_time = time.time()

    # 清空输出目录
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"已清空输出目录: {output_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入文件列表
    input_files = []
    if os.path.isfile(input_path):
        if input_path.endswith('.splat'):
            input_files = [input_path]
        else:
            print(f"错误: {input_path} 不是splat文件")
            return
    elif os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.splat')]
    else:
        print(f"错误: {input_path} 不存在")
        return

    if not input_files:
        print("未找到splat文件")
        return

    print(f"找到 {len(input_files)} 个splat文件")
    print(f"切片模式: {mode}")

    # 根据模式进行不同的初始化
    tile_manager = None
    if mode == "count":
        print(f"每个瓦片点数: {points_per_tile}")
        # 估算总瓦片数
        total_points = sum(get_point_num(f) for f in input_files)
        estimated_tiles = (total_points + points_per_tile - 1) // points_per_tile
        print(f"总点数: {total_points}")
        print(f"估算瓦片数: {estimated_tiles}")
    elif mode == "grid":
        # 计算或使用指定的边界
        if auto_bounds:
            bounds = calculate_bounds_from_files(input_files)

        # 初始化瓦片管理器
        tile_manager = SimpleTileManager(grid_size, bounds)
        print(f"网格大小: {grid_size}")
        print(tile_manager.get_bounds_info())

        # 估算瓦片数量
        if bounds:
            min_x, min_y, min_z, max_x, max_y, max_z = bounds
            tiles_x = int((max_x - min_x) / grid_size) + 1
            tiles_y = int((max_y - min_y) / grid_size) + 1
            tiles_z = int((max_z - min_z) / grid_size) + 1
            estimated_tiles = tiles_x * tiles_y * tiles_z
            print(f"估算最大瓦片数: {tiles_x} x {tiles_y} x {tiles_z} = {estimated_tiles}")
    else:
        print(f"错误: 不支持的切片模式: {mode}")
        return

    # 初始化进度管理
    manager = Manager()
    progress_queue = manager.Queue()

    # 如果是count模式，初始化共享的瓦片信息字典
    shared_tile_info = None
    if mode == "count":
        shared_tile_info = manager.dict()

    # 初始化进度条
    file_num = len(input_files)
    pbar = tqdm(total=file_num, desc="切片处理", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理切片
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for input_file in input_files:
            if mode == "count":
                tasks.append(pool.apply_async(split_splat_file_by_count, (input_file, output_dir, points_per_tile, progress_queue, shared_tile_info)))
            elif mode == "grid":
                tasks.append(pool.apply_async(split_splat_file_by_grid, (input_file, output_dir, tile_manager, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        while completed_tasks < file_num:
            progress_update = progress_queue.get()  # 等待子进程通知进度

            if progress_update is None:
                completed_tasks += 1  # 任务完成信号
                pbar.update(1)
            else:
                pbar.update(progress_update)  # 更新进度条

        # 等待所有任务完成
        for task in tasks:
            task.get()

    # 关闭进度条
    pbar.close()

    # 统计输出文件
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.splat')]

    # 如果是count模式，生成JSON文件
    if mode == "count" and shared_tile_info:
        generate_tiles_json(shared_tile_info, output_dir, input_files)

    end_time = time.time()
    print(f"\n切片完成!")
    print(f"输入文件: {len(input_files)} 个")
    print(f"输出瓦片: {len(output_files)} 个")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    print(f"输出目录: {output_dir}")

    if mode == "count":
        json_file = os.path.join(output_dir, "tiles_info.json")
        if os.path.exists(json_file):
            print(f"瓦片信息文件: {json_file}")


def generate_tiles_json(shared_tile_info: Dict, output_dir: str, input_files: List[str]) -> None:
    """
    生成包含所有瓦片信息的JSON文件

    Args:
        shared_tile_info: 共享的瓦片信息字典
        output_dir: 输出目录
        input_files: 输入文件列表
    """
    import numpy as np

    # 转换共享字典为普通字典
    tiles_info = dict(shared_tile_info)

    # 计算全局边界框
    all_mins = []
    all_maxs = []
    total_points = 0

    for tile_info in tiles_info.values():
        if tile_info["count"] > 0:
            all_mins.append(tile_info["bounding_box"]["min"])
            all_maxs.append(tile_info["bounding_box"]["max"])
            total_points += tile_info["count"]

    if all_mins and all_maxs:
        global_min = np.min(all_mins, axis=0).tolist()
        global_max = np.max(all_maxs, axis=0).tolist()
        global_center = ((np.array(global_min) + np.array(global_max)) / 2).tolist()
        global_size = (np.array(global_max) - np.array(global_min)).tolist()
    else:
        global_min = global_max = global_center = global_size = [0.0, 0.0, 0.0]

    # 构建JSON数据
    json_data = {
        "metadata": {
            "version": "1.0",
            "generator": "Splat Slicer",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_files": [os.path.basename(f) for f in input_files],
            "total_tiles": len(tiles_info),
            "total_points": total_points
        },
        "global_bounding_box": {
            "min": global_min,
            "max": global_max,
            "center": global_center,
            "size": global_size
        },
        "tiles": tiles_info
    }

    # 写入JSON文件
    json_file = os.path.join(output_dir, "tiles_info.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"已生成瓦片信息文件: {json_file}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="Splat文件切片工具 - 将splat文件切分为多个小文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 按点数切片（默认模式），每个瓦片4096个点
  python splat_slicer.py -i input.splat -o output_tiles/ -c 4096

  # 按点数切片整个目录，每个瓦片2048个点
  python splat_slicer.py -i input_dir/ -o output_tiles/ -c 2048

  # 按网格切片，网格大小10.0
  python splat_slicer.py -i input.splat -o output_tiles/ --mode grid -g 10.0

  # 按网格切片，手动指定空间边界
  python splat_slicer.py -i input.splat -o output_tiles/ --mode grid -g 10.0 --bounds -100 -100 -10 100 100 10
        """
    )

    parser.add_argument("-i", "--input", required=True,
                       help="输入splat文件或包含splat文件的目录")
    parser.add_argument("-o", "--output", required=True,
                       help="输出目录，切片后的文件将保存在此目录")
    parser.add_argument("--mode", choices=["count", "grid"], default="count",
                       help="切片模式: count=按点数切片(默认), grid=按网格切片")
    parser.add_argument("-c", "--count", type=int, default=4096,
                       help="每个瓦片的点数（count模式），默认4096")
    parser.add_argument("-g", "--grid-size", type=float, default=10.0,
                       help="网格大小（grid模式，每个瓦片的边长），默认10.0")
    parser.add_argument("--bounds", nargs=6, type=float, metavar=('min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z'),
                       help="手动指定空间边界（grid模式），格式: min_x min_y min_z max_x max_y max_z")
    parser.add_argument("--version", action="version", version="Splat Slicer 1.0")

    args = parser.parse_args()

    # 解析参数
    input_path = args.input
    output_dir = args.output
    mode = args.mode
    points_per_tile = args.count
    grid_size = args.grid_size
    bounds = tuple(args.bounds) if args.bounds else None
    auto_bounds = bounds is None

    # 验证输入
    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在: {input_path}")
        return 1

    if grid_size <= 0:
        print(f"错误: 网格大小必须大于0: {grid_size}")
        return 1

    # 执行切片
    try:
        main_split_splat_files(input_path, output_dir, mode, points_per_tile, grid_size, auto_bounds, bounds)
        return 0
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Windows多进程支持
    exit(main())
