from multiprocessing import Pool, cpu_count, Manager
import os
import time
import struct
from typing import Dict, Tuple
from collections import defaultdict

from tqdm import tqdm
from common import get_point_num, getPointSize, read_ply_file, write_gaussian_file
from tile import TileId
from tile_manager import TileManager
from point import Point

point_num_per_update = 1000

# 将单个高斯点的数据写入分块文件
def write_point_to_tile_file(point_data, tile_id: TileId, tile_files: Dict, output_dir: str, input_id: int = 0, file_format: str = "splat"):
    """
    写入点数据到瓦片文件
    :param point_data: 点数据 (二进制格式用于splat，Point对象用于ply)
    :param tile_id: 瓦片ID
    :param tile_files: 瓦片文件字典
    :param output_dir: 输出目录
    :param input_id: 输入文件ID
    :param file_format: 文件格式 ("splat" 或 "ply")
    """
    tile_key = (tile_id, file_format)
    write_file = tile_files.get(tile_key)

    if not write_file:
        if file_format == "ply":
            ext = f".{input_id}.ply"
            output_file = tile_id.getFilePath(output_dir, ext)
            # 对于PLY格式，我们需要收集点数据而不是直接写入
            tile_files[tile_key] = []
        else:
            ext = f".{input_id}.splat"
            output_file = tile_id.getFilePath(output_dir, ext)
            write_file = open(output_file, "w+b")
            tile_files[tile_key] = write_file

    if file_format == "ply":
        # 对于PLY格式，收集Point对象
        tile_files[tile_key].append(point_data)
    else:
        # 对于SPLAT格式，直接写入二进制数据
        tile_files[tile_key].write(point_data)

def finalize_tile_files(tile_files: Dict, output_dir: str, file_format: str):
    """
    完成瓦片文件的写入
    :param tile_files: 瓦片文件字典
    :param output_dir: 输出目录
    :param file_format: 文件格式
    """
    for tile_key, file_data in tile_files.items():
        if file_format == "ply":
            # tile_key 是 (tile_id, format) 的元组
            tile_id, _ = tile_key
            points = file_data  # 对于PLY格式，这是Point对象列表
            if points:  # 只有当有点数据时才写入文件
                # 从第一个文件的hash生成文件名
                input_id = hash(str(tile_id))  # 简化的ID生成
                ext = f".{abs(input_id)}.ply"
                output_file = tile_id.getFilePath(output_dir, ext)
                write_gaussian_file(output_file, points)
        else:
            # 对于SPLAT格式，关闭文件句柄
            if hasattr(file_data, 'close'):
                file_data.close()

def split_ply_to_tiles(input_file: str, tile_files: Dict, tile_manager: TileManager,
                      output_dir: str, input_id: int, point_num: int, progress_queue, file_format: str = "ply") -> None:
    """
    将PLY文件切分到瓦片中
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("需要安装 plyfile 库来支持PLY格式: pip install plyfile")

    plydata = PlyData.read(input_file)
    vertex = plydata['vertex']

    point_i = 0
    point_update = 0

    for i in range(len(vertex)):
        # 读取位置信息
        position = (float(vertex['x'][i]), float(vertex['y'][i]), float(vertex['z'][i]))

        # 获取瓦片ID
        tile_id = tile_manager.getTileId(position)

        # 创建Point对象
        point = create_point_from_ply_vertex(vertex, i)

        # 根据格式决定数据类型
        if file_format == "ply":
            point_data = point  # 直接传递Point对象
        else:
            point_data = point.to_bytes()  # 转换为二进制数据

        # 写入瓦片文件
        write_point_to_tile_file(point_data, tile_id, tile_files, output_dir, input_id, file_format)

        point_i += 1

        # 每隔1000个点通知主进程一次
        if point_i % point_num_per_update == 0 or point_i == point_num - 1:
            progress_update = (point_i - point_update) / point_num
            progress_queue.put(progress_update)
            point_update = point_i

def create_point_from_ply_vertex(vertex, index: int) -> Point:
    """
    从PLY顶点数据创建Point对象
    """
    import numpy as np

    # 读取位置
    position = (float(vertex['x'][index]), float(vertex['y'][index]), float(vertex['z'][index]))

    # 读取缩放 (需要exp变换)
    scale = (
        float(np.exp(vertex['scale_0'][index])),
        float(np.exp(vertex['scale_1'][index])),
        float(np.exp(vertex['scale_2'][index]))
    )

    # 读取旋转四元数 (归一化)
    rot_0 = float(vertex['rot_0'][index])
    rot_1 = float(vertex['rot_1'][index])
    rot_2 = float(vertex['rot_2'][index])
    rot_3 = float(vertex['rot_3'][index])

    # 归一化四元数
    norm = np.sqrt(rot_0*rot_0 + rot_1*rot_1 + rot_2*rot_2 + rot_3*rot_3)
    if norm > 0:
        rot_0 /= norm
        rot_1 /= norm
        rot_2 /= norm
        rot_3 /= norm

    # 转换为0-255范围的uint8 (w, x, y, z)
    rotation = (
        int((rot_0 + 1.0) * 127.5),  # w
        int((rot_1 + 1.0) * 127.5),  # x
        int((rot_2 + 1.0) * 127.5),  # y
        int((rot_3 + 1.0) * 127.5)   # z
    )

    # 读取透明度 (需要sigmoid变换)
    opacity_raw = float(vertex['opacity'][index])
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid

    # 读取颜色 (从球谐系数的DC分量)
    dc_0 = float(vertex['f_dc_0'][index])
    dc_1 = float(vertex['f_dc_1'][index])
    dc_2 = float(vertex['f_dc_2'][index])

    # 将球谐系数转换为RGB颜色 (0-255范围)
    SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi))
    r = (dc_0 * SH_C0 + 0.5) * 255
    g = (dc_1 * SH_C0 + 0.5) * 255
    b = (dc_2 * SH_C0 + 0.5) * 255

    # 确保颜色值在有效范围内
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    a = max(0, min(255, int(opacity * 255)))

    color = (r, g, b, a)

    return Point(position, color, scale, rotation)


# 将单个高斯溅射的数据文件切块
def split_to_tiles_file(input_file: str, output_dir: str,
                        tile_manager: TileManager, progress_queue) -> None:

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_id = hash(input_file)
    point_num = get_point_num(input_file)
    point_i = 0
    point_update = 0

    tile_files = defaultdict(list)

    # 确定文件格式
    file_format = "ply" if input_file.lower().endswith('.ply') else "splat"

    # 检查文件格式并选择相应的处理方式
    if file_format == "ply":
        # 处理PLY文件
        split_ply_to_tiles(input_file, tile_files, tile_manager, output_dir, input_id,
                          point_num, progress_queue, file_format)
    else:
        # 处理SPLAT文件 (原有逻辑)
        point_size = getPointSize()
        with open(input_file, 'rb') as f:
            while point_i < point_num:
                point_data = f.read(point_size)  # 3个 Float32，每个4字节
                if not point_data:
                    break

                position_data = point_data[0:12]
                position = struct.unpack('3f', position_data)
                tile_id = tile_manager.getTileId(position)
                write_point_to_tile_file(point_data, tile_id, tile_files, output_dir, input_id, file_format)

                point_i += 1

                # 每隔1000个点通知主进程一次
                if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                    progress_update = (point_i - point_update) / point_num
                    progress_queue.put(progress_update)
                    point_update = point_i

    # 关闭文件或写入PLY文件
    finalize_tile_files(tile_files, output_dir, file_format)

    # 通知主进程任务完成
    progress_queue.put(None)  # 使用 None 作为任务完成的信号


# 将高斯溅射的数据切块
def main_split_to_tiles(input_dir: str, output_dir: str,
                        enu_origin: Tuple[float, float] = (0.0, 0.0),
                        tile_zoom: int = 20):

    # 删除目录中的所有文件和子目录
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"目录 {output_dir} 已被清空。")
    else:
        print(f"目录 {output_dir} 不存在。")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有高斯文件 (支持 .splat 和 .ply 格式)
    gaussian_files = [f for f in os.listdir(input_dir) if f.endswith('.splat') or f.endswith('.ply')]
    file_num = len(gaussian_files)
    splat_count = len([f for f in gaussian_files if f.endswith('.splat')])
    ply_count = len([f for f in gaussian_files if f.endswith('.ply')])
    print(f"Found {file_num} gaussian files: {splat_count} .splat files, {ply_count} .ply files.")

    # 初始化进度队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化瓦片管理器
    tile_manager = TileManager(enu_origin, tile_zoom)


    # 初始化进度条
    pbar = tqdm(total=file_num, desc="Splitting files", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理切块
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for gaussian_file in gaussian_files:
            file_path = os.path.join(input_dir, gaussian_file)
            tasks.append(pool.apply_async(split_to_tiles_file, (file_path, output_dir, tile_manager, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        while completed_tasks < file_num:
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
