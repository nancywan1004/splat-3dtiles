"""
将 3D Gaussian Splatting 点云转换为 Cesium 3D Tiles 格式。
其中 gltf 文件包含 KHR_gaussian_splatting 扩展。

参考资料
https://github.com/CesiumGS/glTF/tree/proposal-KHR_gaussian_splatting/extensions/2.0/Khronos/KHR_gaussian_splatting

作者：杨建顺 20250528

"""

import argparse
from multiprocessing import freeze_support
import os
from common import getVersion, get_point_num

from main_convert_to_3dtiles import main_convert_to_3dtiles
from main_convert_to_gltf import main_convert_to_gltf
from main_split_to_tiles import main_split_to_tiles
from main_clean_tiles import main_clean_tiles
from main_build_lod_tiles import main_build_lod_tiles
from main_build_fine_lod import main_build_fine_lod_tiles


def analyze_tile_complexity(split_output_dir):
    """
    分析瓦片文件的复杂度，决定是否使用优化版本
    返回: (should_use_optimized, total_points, total_files, analysis_info)
    """
    print("正在分析瓦片文件复杂度...")

    # 查找所有瓦片文件
    tile_files = []
    for root, dirs, files in os.walk(split_output_dir):
        for file in files:
            if file.endswith('.splat') or file.endswith('.ply'):
                tile_files.append(os.path.join(root, file))

    if not tile_files:
        return False, 0, 0, "未找到瓦片文件"

    total_points = 0
    total_size = 0
    large_files = 0
    ply_files = 0

    print(f"找到 {len(tile_files)} 个瓦片文件，正在分析...")

    # 分析前几个文件来估算
    sample_size = min(10, len(tile_files))
    sample_points = 0
    sample_files = 0

    for idx, file_path in enumerate(tile_files[:sample_size]):
        print(f"  分析样本 {idx + 1}/{sample_size}: {os.path.basename(file_path)}")
        try:
            file_size = os.path.getsize(file_path)
            total_size += file_size

            if file_path.endswith('.ply'):
                ply_files += 1

            # 获取点数量
            point_count = get_point_num(file_path)
            sample_points += point_count
            sample_files += 1

            print(f"    点数: {point_count:,}, 大小: {file_size / (1024*1024):.1f}MB")

            # 检查是否为大文件
            if point_count > 100000 or file_size > 50 * 1024 * 1024:  # 10万个点或50MB
                large_files += 1
                print(f"    ⚠️  检测到大文件")

        except Exception as e:
            print(f"    ✗ 分析失败: {e}")
            continue

    # 估算总点数
    if sample_files > 0:
        avg_points_per_file = sample_points / sample_files
        estimated_total_points = int(avg_points_per_file * len(tile_files))
    else:
        estimated_total_points = 0

    # 决策逻辑
    should_use_optimized = False
    reasons = []

    # 条件1: 总点数超过阈值
    if estimated_total_points > 1000000:  # 超过100万个点
        should_use_optimized = True
        reasons.append(f"总点数过多 ({estimated_total_points:,})")

    # 条件2: 有大文件
    if large_files > 0:
        should_use_optimized = True
        reasons.append(f"包含 {large_files} 个大文件")

    # 条件3: PLY文件比例高
    if ply_files > len(tile_files) * 0.5:  # 超过50%是PLY文件
        should_use_optimized = True
        reasons.append(f"PLY文件比例高 ({ply_files}/{len(tile_files)})")

    # 条件4: 文件总大小
    if total_size > 500 * 1024 * 1024:  # 超过500MB
        should_use_optimized = True
        reasons.append(f"文件总大小过大 ({total_size / (1024*1024):.1f}MB)")

    analysis_info = {
        'total_files': len(tile_files),
        'estimated_points': estimated_total_points,
        'total_size_mb': total_size / (1024*1024),
        'large_files': large_files,
        'ply_files': ply_files,
        'reasons': reasons
    }

    return should_use_optimized, estimated_total_points, len(tile_files), analysis_info


def run_optimized_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis):
    """
    运行优化版本的clean tiles
    """
    print("使用优化版本的clean tiles处理...")

    try:
        # 导入优化版本的函数
        import subprocess
        import sys

        # 构建命令
        cmd = [
            sys.executable, 'main_clean_tiles_optimized.py',
            '--input', split_output_dir,
            '--output', clean_output_dir,
            '--min_alpha', str(min_alpha),
            '--max_scale', str(max_scale),
            '--flyers_num', str(flyers_num),
            '--flyers_dis', str(flyers_dis)
        ]

        print(f"执行命令: {' '.join(cmd)}")

        # 运行优化版本
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ 优化版本clean tiles执行成功")
            return True
        else:
            print(f"✗ 优化版本clean tiles执行失败: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ 优化版本clean tiles执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ 运行优化版本时发生错误: {e}")
        return False


# 主函数
if __name__ == "__main__":
    freeze_support()

    __version__ = getVersion()
    print(f"splat-3dtiles: {__version__}")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将 3D Gaussian Splatting 点云转换为 Cesium 3D Tiles 格式")
    parser.add_argument("--input", "-i", required=True, help="输入的高斯点云文件或文件夹。支持单个 .splat/.ply 文件或包含多个文件的文件夹。")
    parser.add_argument("--output", "-o", required=True, help="输出保存 3dtiles 文件夹.")
    parser.add_argument("--enu_origin", nargs=2, type=float, metavar=('lon', 'lat'), help="指定 ENU 坐标系的原点经纬度 (lon, lat)。默认为 (0.0, 0.0)。")
    parser.add_argument("--tile_zoom", type=int, default=20, help="分块的等级，默认为 20。")
    parser.add_argument("--tile_resolution", type=float, default=0.1, help="用于生成 Lod 的参数，20级代表的精度，默认为 0.1 米。")
    parser.add_argument("--tile_error", type=float, default=1, help="用于生成 tilejson 的 geometric_error 参数，20级代表的误差，默认为 1 米。")
    parser.add_argument("--lod_factor", type=float, default=1.3, help="LOD因子，控制聚类激进程度。值越小越精细(1.2-2.0)，默认1.3。")
    parser.add_argument("--use_fine_lod", action="store_true", help="使用精细LOD算法，适合大规模点云场景。")
    parser.add_argument("--target_reduction_ratio", type=float, default=0.65, help="精细LOD的目标减少比例(0.5-0.8)，默认0.65。")
    parser.add_argument("--lod_levels", type=int, default=4, help="LOD层数，默认为4层。")
    parser.add_argument("--lod_skip", type=int, default=0, help="LOD跳级参数，0表示每级都生成，1表示跳1级，默认为0。")


    parser.add_argument("--min_alpha", type=float, default=1.0, help="最小透明度阈值，小于该阈值的高斯点会被过滤，默认为 1.0。")
    parser.add_argument("--max_scale", type=float, default=10000, help="最大缩放值阈值，大于该阈值的高斯点会被过滤，默认为 10000。")
    parser.add_argument("--flyers_num", type=int, default=25, help="移除飞点的最临近点数，默认为25。")
    parser.add_argument("--flyers_dis", type=float, default=10, help="移除飞点的距离因子，最小移除的越多，默认为10。")

    # 优化选项
    parser.add_argument("--force_optimized", action="store_true", help="强制使用优化版本的clean tiles处理。")
    parser.add_argument("--force_standard", action="store_true", help="强制使用标准版本的clean tiles处理。")

    # 跳过步骤选项
    parser.add_argument("--skip_split", action="store_true", help="跳过split步骤，直接从输入目录的瓦片文件开始构建LOD。")
    parser.add_argument("--skip_clean", action="store_true", help="跳过clean步骤，直接从输入目录开始构建LOD。")

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    enu_origin = (args.enu_origin[0], args.enu_origin[1]) if args.enu_origin else (0.0, 0.0)
    tile_zoom = args.tile_zoom
    tile_resolution = args.tile_resolution
    tile_error = args.tile_error
    lod_factor = args.lod_factor
    use_fine_lod = args.use_fine_lod
    target_reduction_ratio = args.target_reduction_ratio
    lod_levels = args.lod_levels
    lod_skip = args.lod_skip

    min_alpha = args.min_alpha
    max_scale = args.max_scale
    flyers_num = args.flyers_num
    flyers_dis = args.flyers_dis


    split_output_dir = os.path.join(output_dir, f"split")
    build_output_dir = os.path.join(output_dir, f"build")
    result_output_dir = os.path.join(output_dir, f"result")

    clean_output_dir = os.path.join(build_output_dir, f"{tile_zoom}")

    # 检查是否跳过split步骤
    if args.skip_split:
        print(f"🚀 跳过split步骤，检查split输出目录...")

        # 检查split输出目录是否存在且包含瓦片文件
        if os.path.exists(split_output_dir):
            tile_files = []
            for root, dirs, files in os.walk(split_output_dir):
                for file in files:
                    if file.endswith('.ply') or file.endswith('.splat'):
                        tile_files.append(os.path.join(root, file))

            if tile_files:
                print(f"📁 在split目录中找到 {len(tile_files)} 个瓦片文件，直接使用")
            else:
                print(f"⚠️  split目录存在但为空，需要执行split操作")
                print(f"----main_split_to_tiles start:[{tile_zoom}][{input_path}][{split_output_dir}]")
                main_split_to_tiles(input_path, split_output_dir, enu_origin, tile_zoom)
        else:
            print(f"⚠️  split目录不存在，需要执行split操作")
            print(f"----main_split_to_tiles start:[{tile_zoom}][{input_path}][{split_output_dir}]")
            main_split_to_tiles(input_path, split_output_dir, enu_origin, tile_zoom)
    else:
        print(f"----main_split_to_tiles start:[{tile_zoom}][{input_path}][{split_output_dir}]")
        main_split_to_tiles(input_path, split_output_dir, enu_origin, tile_zoom)

    # 检查是否跳过clean步骤
    if args.skip_clean:
        print(f"🚀 跳过clean步骤，直接使用split输出目录")
        clean_output_dir = split_output_dir
        need_clean = False
    else:
        print(f"----main_clean_tiles start:[{tile_zoom}][{split_output_dir}][{clean_output_dir}]")
        need_clean = True

    # 如果需要执行clean操作
    if need_clean:
        # 检查用户强制选择
        if args.force_optimized and args.force_standard:
            print("⚠️  警告: 不能同时指定 --force_optimized 和 --force_standard，将使用自动检测")
            force_choice = None
        elif args.force_optimized:
            force_choice = "optimized"
            print("🔧 用户强制选择: 使用优化版本")
        elif args.force_standard:
            force_choice = "standard"
            print("🔧 用户强制选择: 使用标准版本")
        else:
            force_choice = None

        # 如果没有强制选择，进行自动分析
        if force_choice is None:
            should_use_optimized, estimated_points, total_files, analysis_info = analyze_tile_complexity(split_output_dir)

            print(f"瓦片分析结果:")
            print(f"  文件数量: {analysis_info['total_files']}")
            print(f"  估算点数: {analysis_info['estimated_points']:,}")
            print(f"  文件总大小: {analysis_info['total_size_mb']:.1f}MB")
            print(f"  大文件数量: {analysis_info['large_files']}")
            print(f"  PLY文件数量: {analysis_info['ply_files']}")

            if should_use_optimized:
                print(f"🚀 检测到复杂场景，自动使用优化版本:")
                for reason in analysis_info['reasons']:
                    print(f"   - {reason}")
                force_choice = "optimized"
            else:
                print("📋 场景复杂度适中，使用标准版本")
                force_choice = "standard"

        # 执行相应的版本
        if force_choice == "optimized":
            print("执行优化版本clean tiles...")
            success = run_optimized_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis)

            if not success:
                print("⚠️  优化版本执行失败，回退到标准版本...")
                main_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis)
        else:
            print("执行标准版本clean tiles...")
            main_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis)


    # 计算LOD层级序列，从初始层级开始应用跳级
    lod_levels_to_generate = []
    current_zoom = tile_zoom - (1 + lod_skip)  # 第一个LOD层级就应用跳级

    for i in range(lod_levels):
        if current_zoom > 0:
            lod_levels_to_generate.append(current_zoom)
            current_zoom -= (1 + lod_skip)  # 后续层级继续应用跳级
        else:
            break

    print(f"开始生成LOD，总共将生成 {len(lod_levels_to_generate)} 层，层级序列: {lod_levels_to_generate}，跳级参数: {lod_skip}")

    lod_input_dir = clean_output_dir

    for i, lod_zoom in enumerate(lod_levels_to_generate):
        lod_output_dir = os.path.join(build_output_dir, f"{lod_zoom}")

        if use_fine_lod:
            print(f"----main_build_fine_lod_tiles start:[{lod_zoom}][{lod_input_dir}][{lod_output_dir}] (第{i + 1}层)")
            main_build_fine_lod_tiles(lod_input_dir, lod_output_dir, enu_origin, lod_zoom,
                                    tile_resolution, target_reduction_ratio)
        else:
            print(f"----main_build_lod_tiles start:[{lod_zoom}][{lod_input_dir}][{lod_output_dir}] (第{i + 1}层)")
            main_build_lod_tiles(lod_input_dir, lod_output_dir, enu_origin, lod_zoom, tile_resolution, lod_factor)

        lod_input_dir = lod_output_dir

    # main_convert_to_3dtiles(build_output_dir, result_output_dir, enu_origin, tile_zoom, tile_error)
