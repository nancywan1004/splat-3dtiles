
__version__ = '0.1'
import os
import struct
from typing import List
import numpy as np

from point import Point


point_size = 3*4 + 3*4 + 4*1 + 4*1  # 3 float32 + 3 float32 + 4 uint8 + 4 uint8 = 32字节

def getVersion():
    return __version__

def getPointSize():
    return point_size


def get_point_num(file_path: str) -> int:
    """
    获取文件中的点数量，支持 .splat 和 .ply 格式
    """
    if file_path.lower().endswith('.ply'):
        return get_ply_point_num(file_path)
    else:
        # 默认按 splat 格式处理
        stats = os.stat(file_path)
        point_size = getPointSize()
        point_num = stats.st_size // point_size
        return point_num

def get_ply_point_num(file_path: str) -> int:
    """
    获取PLY文件中的点数量
    """
    try:
        from plyfile import PlyData
        plydata = PlyData.read(file_path)
        return len(plydata['vertex'])
    except ImportError:
        raise ImportError("需要安装 plyfile 库来支持PLY格式: pip install plyfile")
    except Exception as e:
        raise Exception(f"读取PLY文件失败: {e}")

# 读取数据
def read_splat_file(file_path: str) -> List[Point]:
    """
    读取二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :return: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    points = []
    with open(file_path, 'rb') as f:
        while True:
            position_data = f.read(3 * 4)  # 3个 Float32，每个4字节
            if not position_data:
                break
            position = struct.unpack('3f', position_data)
            scale = struct.unpack('3f', f.read(3 * 4))
            color = struct.unpack('4B', f.read(4 * 1))
            rotation = struct.unpack('4B', f.read(4 * 1))

            # 调整四元数顺序 (x, y, z, w) -> (w, x, y, z)
            # rotation = (rotation[1], rotation[2], rotation[3], rotation[0])

            points.append(Point(position, color, scale, rotation))
    return points

def read_ply_file(file_path: str) -> List[Point]:
    """
    读取PLY格式的3D Gaussian Splatting文件
    :param file_path: PLY文件路径
    :return: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("需要安装 plyfile 库来支持PLY格式: pip install plyfile")

    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']

    points = []

    for i in range(len(vertex)):
        # 读取位置
        position = (float(vertex['x'][i]), float(vertex['y'][i]), float(vertex['z'][i]))

        # 读取缩放 (需要exp变换)
        scale = (
            float(np.exp(vertex['scale_0'][i])),
            float(np.exp(vertex['scale_1'][i])),
            float(np.exp(vertex['scale_2'][i]))
        )

        # 读取旋转四元数 (归一化)
        rot_0 = float(vertex['rot_0'][i])
        rot_1 = float(vertex['rot_1'][i])
        rot_2 = float(vertex['rot_2'][i])
        rot_3 = float(vertex['rot_3'][i])

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
        opacity_raw = float(vertex['opacity'][i])
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid

        # 读取球谐系数 - 动态适配不同阶数
        sh_coeffs = []

        # DC分量 (0阶球谐系数) - 必须存在
        try:
            dc_0 = float(vertex['f_dc_0'][i])
            dc_1 = float(vertex['f_dc_1'][i])
            dc_2 = float(vertex['f_dc_2'][i])
            sh_coeffs.extend([dc_0, dc_1, dc_2])
        except (ValueError, KeyError) as e:
            # 如果连DC分量都没有，使用默认值
            print(f"警告: 无法读取DC分量，使用默认值: {e}")
            sh_coeffs.extend([0.0, 0.0, 0.0])

        # 动态检测并读取高阶球谐系数
        # 首先检查文件中实际存在哪些f_rest字段
        available_rest_fields = []

        # 从PLY元素的properties中获取字段名
        field_names = [prop.name for prop in vertex.properties]

        for field_name in field_names:
            if field_name.startswith('f_rest_'):
                try:
                    rest_index = int(field_name.split('_')[2])
                    available_rest_fields.append(rest_index)
                except (ValueError, IndexError):
                    continue

        # 按索引排序
        available_rest_fields.sort()

        # 读取存在的高阶球谐系数
        for rest_index in available_rest_fields:
            rest_name = f'f_rest_{rest_index}'
            try:
                sh_coeffs.append(float(vertex[rest_name][i]))
            except (ValueError, KeyError):
                sh_coeffs.append(0.0)

        # 如果需要，填充到标准的48个系数（3个DC + 45个高阶）
        while len(sh_coeffs) < 48:
            sh_coeffs.append(0.0)

        # 将DC分量转换为RGB颜色 (用于向后兼容)
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

        points.append(Point(position, color, scale, rotation, sh_coeffs))

    return points

def read_gaussian_file(file_path: str) -> List[Point]:
    """
    通用的高斯文件读取函数，自动检测文件格式
    :param file_path: 文件路径 (.splat 或 .ply)
    :return: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    if file_path.lower().endswith('.ply'):
        return read_ply_file(file_path)
    else:
        # 默认按 splat 格式处理
        return read_splat_file(file_path)

# 写入数据
def write_splat_file(file_path: str, points: List[Point]):
    """
    将 Point 对象列表写入二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :param points: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    with open(file_path, 'wb') as f:
        for point in points:
            # 写入位置 (3个 Float32)
            f.write(struct.pack('3f', *point.position))
            # 写入缩放 (3个 Float32)
            f.write(struct.pack('3f', *point.scale))
            # 写入颜色 (4个 Byte)
            f.write(struct.pack('4B', *point.color))
            # 写入旋转 (4个 Byte)，调整四元数顺序 (w, x, y, z) -> (x, y, z, w)
            # rotation = (point.rotation[1], point.rotation[2], point.rotation[3], point.rotation[0])
            f.write(struct.pack('4B', *point.rotation))

def write_ply_file(file_path: str, points: List[Point]):
    """
    将 Point 对象列表写入PLY格式文件
    :param file_path: PLY文件路径
    :param points: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise ImportError("需要安装 plyfile 库来支持PLY格式: pip install plyfile")

    # 检测需要的球谐系数阶数
    max_sh_coeffs = 3  # 至少需要DC分量
    if points:
        for point in points:
            if hasattr(point, 'sh_coeffs') and point.sh_coeffs:
                max_sh_coeffs = max(max_sh_coeffs, len(point.sh_coeffs))

    # 计算需要的f_rest字段数量
    rest_fields_count = max(0, max_sh_coeffs - 3)

    if not points:
        # 创建空的PLY文件，使用动态的球谐系数结构
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
        ]
        # 只添加需要的高阶球谐系数字段
        for j in range(rest_fields_count):
            dtype_list.append((f'f_rest_{j}', 'f4'))

        vertex = np.array([], dtype=dtype_list)
    else:
        # 准备数据数组
        num_points = len(points)
        vertex_data = []

        for point in points:
            # 位置
            x, y, z = point.position

            # 缩放 (转换回对数空间)
            scale_0 = float(np.log(max(point.scale[0], 1e-8)))
            scale_1 = float(np.log(max(point.scale[1], 1e-8)))
            scale_2 = float(np.log(max(point.scale[2], 1e-8)))

            # 旋转四元数 (从0-255范围转换回-1到1范围)
            rot_w = (point.rotation[0] / 127.5) - 1.0
            rot_x = (point.rotation[1] / 127.5) - 1.0
            rot_y = (point.rotation[2] / 127.5) - 1.0
            rot_z = (point.rotation[3] / 127.5) - 1.0

            # 归一化四元数
            norm = np.sqrt(rot_w*rot_w + rot_x*rot_x + rot_y*rot_y + rot_z*rot_z)
            if norm > 0:
                rot_w /= norm
                rot_x /= norm
                rot_y /= norm
                rot_z /= norm

            # 透明度 (从0-255转换为logit空间)
            alpha = point.color[3] / 255.0
            alpha = max(1e-8, min(1.0 - 1e-8, alpha))  # 避免数值问题
            opacity = float(np.log(alpha / (1.0 - alpha)))  # logit变换

            # 处理球谐系数
            if hasattr(point, 'sh_coeffs') and len(point.sh_coeffs) >= 3:
                # 使用保存的球谐系数
                sh_coeffs = point.sh_coeffs[:]  # 复制一份
                f_dc_0, f_dc_1, f_dc_2 = sh_coeffs[0], sh_coeffs[1], sh_coeffs[2]
                # 获取高阶系数，只取需要的数量
                f_rest = sh_coeffs[3:3+rest_fields_count] if len(sh_coeffs) > 3 else []
                # 如果高阶系数不够，用0填充
                while len(f_rest) < rest_fields_count:
                    f_rest.append(0.0)
            else:
                # 从RGB颜色转换为球谐系数DC分量
                SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi))
                f_dc_0 = (point.color[0] / 255.0 - 0.5) / SH_C0
                f_dc_1 = (point.color[1] / 255.0 - 0.5) / SH_C0
                f_dc_2 = (point.color[2] / 255.0 - 0.5) / SH_C0
                f_rest = [0.0] * rest_fields_count  # 高阶系数设为0

            # 构建完整的顶点数据
            vertex_tuple = (
                x, y, z,
                scale_0, scale_1, scale_2,
                rot_w, rot_x, rot_y, rot_z,
                opacity,
                f_dc_0, f_dc_1, f_dc_2
            )
            # 添加需要的高阶球谐系数
            vertex_tuple += tuple(f_rest)

            vertex_data.append(vertex_tuple)

        # 创建动态的数据类型定义，根据实际需要的球谐系数字段
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
        ]
        # 只添加需要的高阶球谐系数字段
        for j in range(rest_fields_count):
            dtype_list.append((f'f_rest_{j}', 'f4'))

        # 创建结构化数组
        vertex = np.array(vertex_data, dtype=dtype_list)

    # 创建PLY元素
    el = PlyElement.describe(vertex, 'vertex')

    # 写入PLY文件
    PlyData([el], text=False).write(file_path)

def write_gaussian_file(file_path: str, points: List[Point]):
    """
    通用的高斯文件写入函数，根据文件扩展名自动选择格式
    :param file_path: 文件路径 (.splat 或 .ply)
    :param points: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    if file_path.lower().endswith('.ply'):
        write_ply_file(file_path, points)
    else:
        # 默认按 splat 格式处理
        write_splat_file(file_path, points)