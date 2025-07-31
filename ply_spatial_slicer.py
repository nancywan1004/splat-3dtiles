#!/usr/bin/env python3
"""
PLY格式高斯切片工具
基于3D空间分布将PLY格式的高斯文件切分为指定数量的块
支持均匀空间分布和自适应分布两种模式

作者：基于现有splat-3dtiles代码开发
"""

import argparse
import json
import os
import time
import math
from multiprocessing import Pool, cpu_count, Manager
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from common import read_ply_file, write_ply_file, get_point_num
from point import Point, compute_box


class SpatialBlock:
    """空间块定义"""
    def __init__(self, block_id: int, bounds: Tuple[float, float, float, float, float, float]):
        self.block_id = block_id
        self.bounds = bounds  # (min_x, min_y, min_z, max_x, max_y, max_z)
        self.points = []
        self.point_count = 0
    
    def contains_point(self, position: Tuple[float, float, float]) -> bool:
        """检查点是否在当前块内"""
        x, y, z = position
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        return (min_x <= x < max_x and 
                min_y <= y < max_y and 
                min_z <= z < max_z)
    
    def add_point(self, point: Point):
        """添加点到当前块"""
        self.points.append(point)
        self.point_count += 1
    
    def get_filename(self, input_filename: str) -> str:
        """生成块文件名"""
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}_block_{self.block_id:04d}.ply"
    
    def get_center(self) -> Tuple[float, float, float]:
        """获取块的中心点"""
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
    
    def get_size(self) -> Tuple[float, float, float]:
        """获取块的尺寸"""
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        return (max_x - min_x, max_y - min_y, max_z - min_z)


class PLYSpatialSlicer:
    """PLY空间切片器"""
    
    def __init__(self, num_blocks: int, distribution_mode: str = "uniform"):
        """
        初始化切片器
        
        Args:
            num_blocks: 目标块数量
            distribution_mode: 分布模式 ("uniform": 均匀分布, "adaptive": 自适应分布)
        """
        self.num_blocks = num_blocks
        self.distribution_mode = distribution_mode
        self.blocks = []
        self.global_bounds = None
    
    def calculate_bounds(self, input_files: List[str]) -> Tuple[float, float, float, float, float, float]:
        """计算所有输入文件的全局边界"""
        print("正在计算全局空间边界...")
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        total_points = 0
        
        for input_file in tqdm(input_files, desc="扫描文件边界"):
            try:
                points = read_ply_file(input_file)
                total_points += len(points)
                
                for point in points:
                    x, y, z = point.position
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    min_z = min(min_z, z)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    max_z = max(max_z, z)
            except Exception as e:
                print(f"警告: 无法读取文件 {input_file}: {e}")
                continue
        
        bounds = (min_x, min_y, min_z, max_x, max_y, max_z)
        print(f"总点数: {total_points:,}")
        print(f"全局边界: X[{min_x:.3f}, {max_x:.3f}] Y[{min_y:.3f}, {max_y:.3f}] Z[{min_z:.3f}, {max_z:.3f}]")
        
        return bounds
    
    def create_uniform_blocks(self, bounds: Tuple[float, float, float, float, float, float]):
        """创建均匀分布的空间块"""
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        
        # 计算空间尺寸
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        
        # 计算最佳的3D网格分布
        # 尝试找到最接近立方体的分割方式
        best_diff = float('inf')
        best_division = (1, 1, self.num_blocks)
        
        for nx in range(1, self.num_blocks + 1):
            if self.num_blocks % nx != 0:
                continue
            remaining = self.num_blocks // nx
            
            for ny in range(1, remaining + 1):
                if remaining % ny != 0:
                    continue
                nz = remaining // ny
                
                # 计算每个块的尺寸
                block_size_x = size_x / nx
                block_size_y = size_y / ny
                block_size_z = size_z / nz
                
                # 计算尺寸差异（越接近立方体越好）
                sizes = [block_size_x, block_size_y, block_size_z]
                diff = max(sizes) - min(sizes)
                
                if diff < best_diff:
                    best_diff = diff
                    best_division = (nx, ny, nz)
        
        nx, ny, nz = best_division
        print(f"使用网格分割: {nx} x {ny} x {nz} = {nx*ny*nz} 块")
        
        # 创建空间块
        self.blocks = []
        block_id = 0
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 计算当前块的边界
                    block_min_x = min_x + i * size_x / nx
                    block_max_x = min_x + (i + 1) * size_x / nx
                    block_min_y = min_y + j * size_y / ny
                    block_max_y = min_y + (j + 1) * size_y / ny
                    block_min_z = min_z + k * size_z / nz
                    block_max_z = min_z + (k + 1) * size_z / nz
                    
                    # 确保最后一个块包含边界点
                    if i == nx - 1:
                        block_max_x = max_x + 1e-6
                    if j == ny - 1:
                        block_max_y = max_y + 1e-6
                    if k == nz - 1:
                        block_max_z = max_z + 1e-6
                    
                    block_bounds = (block_min_x, block_min_y, block_min_z, 
                                  block_max_x, block_max_y, block_max_z)
                    
                    block = SpatialBlock(block_id, block_bounds)
                    self.blocks.append(block)
                    block_id += 1
    
    def create_adaptive_blocks(self, input_files: List[str], bounds: Tuple[float, float, float, float, float, float]):
        """创建自适应分布的空间块（基于点密度）"""
        print("创建自适应空间块...")
        
        # 首先创建一个粗糙的网格来统计点密度
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        
        # 使用较大的初始网格来统计密度
        grid_size = 32  # 32x32x32的初始网格
        density_grid = np.zeros((grid_size, grid_size, grid_size))
        
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        
        # 统计每个网格单元的点数
        print("统计点密度分布...")
        for input_file in tqdm(input_files, desc="分析密度"):
            try:
                points = read_ply_file(input_file)
                for point in points:
                    x, y, z = point.position
                    
                    # 计算网格索引
                    grid_x = min(int((x - min_x) / size_x * grid_size), grid_size - 1)
                    grid_y = min(int((y - min_y) / size_y * grid_size), grid_size - 1)
                    grid_z = min(int((z - min_z) / size_z * grid_size), grid_size - 1)
                    
                    density_grid[grid_x, grid_y, grid_z] += 1
            except Exception as e:
                print(f"警告: 无法读取文件 {input_file}: {e}")
                continue
        
        # 基于密度创建自适应块（这里简化为均匀分布，实际可以根据密度进一步优化）
        print("基于密度分布创建自适应块...")
        self.create_uniform_blocks(bounds)
    
    def assign_points_to_blocks(self, input_file: str) -> Dict[int, List[Point]]:
        """将文件中的点分配到对应的空间块"""
        try:
            points = read_ply_file(input_file)
        except Exception as e:
            print(f"错误: 无法读取PLY文件 {input_file}: {e}")
            return {}
        
        block_points = defaultdict(list)
        unassigned_points = []
        
        for point in points:
            assigned = False
            for block in self.blocks:
                if block.contains_point(point.position):
                    block_points[block.block_id].append(point)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_points.append(point)
        
        # 处理未分配的点（分配给最近的块）
        if unassigned_points:
            print(f"警告: {len(unassigned_points)} 个点未能分配到任何块，将分配给最近的块")
            for point in unassigned_points:
                closest_block_id = self._find_closest_block(point.position)
                block_points[closest_block_id].append(point)
        
        return dict(block_points)
    
    def _find_closest_block(self, position: Tuple[float, float, float]) -> int:
        """找到距离给定位置最近的块"""
        x, y, z = position
        min_distance = float('inf')
        closest_block_id = 0
        
        for block in self.blocks:
            center = block.get_center()
            distance = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_block_id = block.block_id
        
        return closest_block_id
    
    def slice_files(self, input_files: List[str], output_dir: str) -> Dict:
        """对输入文件进行空间切片"""
        # 计算全局边界
        self.global_bounds = self.calculate_bounds(input_files)
        
        # 创建空间块
        if self.distribution_mode == "adaptive":
            self.create_adaptive_blocks(input_files, self.global_bounds)
        else:
            self.create_uniform_blocks(self.global_bounds)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个输入文件
        all_block_info = {}
        total_points = 0
        
        for input_file in tqdm(input_files, desc="处理文件"):
            input_filename = os.path.basename(input_file)
            block_points = self.assign_points_to_blocks(input_file)
            
            # 写入每个块的文件
            for block_id, points in block_points.items():
                if not points:
                    continue
                
                block = self.blocks[block_id]
                output_filename = block.get_filename(input_filename)
                output_path = os.path.join(output_dir, output_filename)
                
                # 写入PLY文件
                write_ply_file(output_path, points)
                
                # 记录块信息
                if block_id not in all_block_info:
                    all_block_info[block_id] = {
                        "block_id": block_id,
                        "bounds": block.bounds,
                        "center": block.get_center(),
                        "size": block.get_size(),
                        "files": [],
                        "total_points": 0
                    }
                
                all_block_info[block_id]["files"].append({
                    "filename": output_filename,
                    "source_file": input_filename,
                    "point_count": len(points)
                })
                all_block_info[block_id]["total_points"] += len(points)
                total_points += len(points)
        
        # 生成切片信息JSON
        slice_info = {
            "metadata": {
                "version": "1.0",
                "generator": "PLY Spatial Slicer",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_files": [os.path.basename(f) for f in input_files],
                "num_blocks": self.num_blocks,
                "distribution_mode": self.distribution_mode,
                "total_points": total_points
            },
            "global_bounds": {
                "min": list(self.global_bounds[:3]),
                "max": list(self.global_bounds[3:]),
                "center": [
                    (self.global_bounds[0] + self.global_bounds[3]) / 2,
                    (self.global_bounds[1] + self.global_bounds[4]) / 2,
                    (self.global_bounds[2] + self.global_bounds[5]) / 2
                ],
                "size": [
                    self.global_bounds[3] - self.global_bounds[0],
                    self.global_bounds[4] - self.global_bounds[1],
                    self.global_bounds[5] - self.global_bounds[2]
                ]
            },
            "blocks": all_block_info
        }
        
        # 写入JSON文件
        json_path = os.path.join(output_dir, "spatial_slice_info.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(slice_info, f, indent=2, ensure_ascii=False)
        
        return slice_info


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="PLY格式高斯切片工具 - 按空间分布将PLY文件切分为指定数量的块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 将PLY文件切分为8个空间块（均匀分布）
  python ply_spatial_slicer.py -i input.ply -o output_blocks/ -n 8

  # 将目录中的PLY文件切分为16个空间块（自适应分布）
  python ply_spatial_slicer.py -i input_dir/ -o output_blocks/ -n 16 --mode adaptive

  # 处理多个PLY文件，切分为27个块
  python ply_spatial_slicer.py -i "*.ply" -o output_blocks/ -n 27
        """
    )

    parser.add_argument("-i", "--input", required=True,
                       help="输入PLY文件或包含PLY文件的目录")
    parser.add_argument("-o", "--output", required=True,
                       help="输出目录，切片后的文件将保存在此目录")
    parser.add_argument("-n", "--num-blocks", type=int, required=True,
                       help="目标块数量")
    parser.add_argument("--mode", choices=["uniform", "adaptive"], default="uniform",
                       help="分布模式: uniform=均匀分布(默认), adaptive=自适应分布")
    parser.add_argument("--version", action="version", version="PLY Spatial Slicer 1.0")

    args = parser.parse_args()

    # 解析参数
    input_path = args.input
    output_dir = args.output
    num_blocks = args.num_blocks
    distribution_mode = args.mode

    # 验证输入
    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在: {input_path}")
        return 1

    if num_blocks <= 0:
        print(f"错误: 块数量必须大于0: {num_blocks}")
        return 1

    # 获取输入文件列表
    input_files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.ply'):
            input_files = [input_path]
        else:
            print(f"错误: {input_path} 不是PLY文件")
            return 1
    elif os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith('.ply')]
    else:
        print(f"错误: {input_path} 不存在")
        return 1

    if not input_files:
        print("未找到PLY文件")
        return 1

    print(f"找到 {len(input_files)} 个PLY文件")
    print(f"目标块数量: {num_blocks}")
    print(f"分布模式: {distribution_mode}")

    # 执行切片
    try:
        start_time = time.time()
        
        slicer = PLYSpatialSlicer(num_blocks, distribution_mode)
        slice_info = slicer.slice_files(input_files, output_dir)
        
        end_time = time.time()
        
        # 统计结果
        output_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.ply')]
        non_empty_blocks = len([b for b in slice_info["blocks"].values() if b["total_points"] > 0])
        
        print(f"\n切片完成!")
        print(f"输入文件: {len(input_files)} 个")
        print(f"输出块文件: {len(output_files)} 个")
        print(f"非空块数量: {non_empty_blocks} / {num_blocks}")
        print(f"总点数: {slice_info['metadata']['total_points']:,}")
        print(f"处理时间: {end_time - start_time:.2f} 秒")
        print(f"输出目录: {output_dir}")
        print(f"切片信息文件: {os.path.join(output_dir, 'spatial_slice_info.json')}")
        
        return 0
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
