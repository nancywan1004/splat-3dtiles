#!/usr/bin/env python3
"""
直接从PLY文件构建LOD的脚本
跳过split和clean步骤，直接开始构建LOD层级
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

from common import getVersion
from main_split_to_tiles import main_split_to_tiles
from main_clean_tiles import main_clean_tiles
from main_build_lod_tiles import main_build_lod_tiles
from main_build_fine_lod import main_build_fine_lod_tiles
from main_convert_to_3dtiles import main_convert_to_3dtiles

def build_lod_from_ply(input_dir, output_dir, enu_origin, tile_zoom, tile_resolution, 
                      tile_error, lod_factor, use_fine_lod, target_reduction_ratio,
                      min_alpha, max_scale, flyers_num, flyers_dis, skip_clean=False):
    """
    从PLY文件直接构建LOD的主函数
    """
    print("=" * 60)
    print("从PLY文件直接构建LOD")
    print("=" * 60)
    
    # 设置目录结构
    split_output_dir = os.path.join(output_dir, "split")
    build_output_dir = os.path.join(output_dir, "build")
    result_output_dir = os.path.join(output_dir, "result")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 查找PLY文件
    ply_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    if not ply_files:
        print(f"❌ 在输入目录中没有找到PLY文件: {input_dir}")
        return False
    
    print(f"📁 找到 {len(ply_files)} 个PLY文件")
    for ply_file in ply_files:
        print(f"   - {os.path.relpath(ply_file, input_dir)}")
    
    try:
        # 步骤1: Split - 将PLY文件分割成瓦片
        print(f"\n🔄 步骤1: 分割PLY文件到瓦片...")
        print(f"   输入: {input_dir}")
        print(f"   输出: {split_output_dir}")
        print(f"   瓦片级别: {tile_zoom}")
        
        main_split_to_tiles(input_dir, split_output_dir, enu_origin, tile_zoom)
        print("✅ 分割完成")
        
        # 步骤2: Clean - 清理瓦片（可选）
        clean_output_dir = os.path.join(build_output_dir, str(tile_zoom))
        
        if skip_clean:
            print(f"\n🚀 跳过清理步骤，直接使用分割输出")
            clean_output_dir = split_output_dir
        else:
            print(f"\n🔄 步骤2: 清理瓦片...")
            print(f"   输入: {split_output_dir}")
            print(f"   输出: {clean_output_dir}")
            
            main_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis)
            print("✅ 清理完成")
        
        # 步骤3: 构建LOD层级
        print(f"\n🔄 步骤3: 构建LOD层级...")
        
        lod_zoom = tile_zoom - 1
        lod_input_dir = clean_output_dir
        lod_levels_built = 0
        
        while lod_zoom > tile_zoom - 6:  # 构建5个LOD层级
            lod_output_dir = os.path.join(build_output_dir, str(lod_zoom))
            
            print(f"   构建LOD级别 {lod_zoom}: {os.path.basename(lod_input_dir)} -> {os.path.basename(lod_output_dir)}")
            
            if use_fine_lod:
                print(f"   使用精细LOD算法，目标减少比例: {target_reduction_ratio}")
                main_build_fine_lod_tiles(lod_input_dir, lod_output_dir, enu_origin, lod_zoom,
                                        tile_resolution, target_reduction_ratio)
            else:
                print(f"   使用标准LOD算法，LOD因子: {lod_factor}")
                main_build_lod_tiles(lod_input_dir, lod_output_dir, enu_origin, lod_zoom, 
                                   tile_resolution, lod_factor)
            
            lod_input_dir = lod_output_dir
            lod_zoom -= 1
            lod_levels_built += 1
            print(f"   ✅ LOD级别 {lod_zoom + 1} 构建完成")
        
        print(f"✅ 所有LOD层级构建完成，共构建 {lod_levels_built} 个层级")
        
        # 步骤4: 转换为3D Tiles
        print(f"\n🔄 步骤4: 转换为3D Tiles...")
        print(f"   输入: {build_output_dir}")
        print(f"   输出: {result_output_dir}")
        
        main_convert_to_3dtiles(build_output_dir, result_output_dir, enu_origin, tile_zoom, tile_error)
        print("✅ 3D Tiles转换完成")
        
        print(f"\n🎉 LOD构建完成！")
        print(f"   输出目录: {result_output_dir}")
        print(f"   瓦片级别: {tile_zoom - lod_levels_built} - {tile_zoom}")
        
        return True
        
    except Exception as e:
        print(f"❌ 构建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    freeze_support()
    
    __version__ = getVersion()
    print(f"splat-3dtiles LOD Builder: {__version__}")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="直接从PLY文件构建LOD层级")
    parser.add_argument("--input", "-i", required=True, help="输入的PLY文件目录")
    parser.add_argument("--output", "-o", required=True, help="输出的3D Tiles目录")
    parser.add_argument("--enu_origin", nargs=2, type=float, metavar=('lon', 'lat'), 
                       help="ENU坐标系原点经纬度 (lon, lat)，默认 (0.0, 0.0)")
    parser.add_argument("--tile_zoom", type=int, default=20, help="瓦片级别，默认20")
    parser.add_argument("--tile_resolution", type=float, default=0.05, 
                       help="瓦片分辨率，默认0.05米")
    parser.add_argument("--tile_error", type=float, default=1, 
                       help="几何误差，默认1米")
    parser.add_argument("--lod_factor", type=float, default=1.2, 
                       help="LOD因子，控制聚类激进程度(1.2-2.0)，默认1.2")
    parser.add_argument("--use_fine_lod", action="store_true", 
                       help="使用精细LOD算法")
    parser.add_argument("--target_reduction_ratio", type=float, default=0.8, 
                       help="精细LOD的目标减少比例(0.5-0.8)，默认0.8")
    
    # Clean参数
    parser.add_argument("--min_alpha", type=float, default=1.0, 
                       help="最小透明度阈值，默认1.0")
    parser.add_argument("--max_scale", type=float, default=10000, 
                       help="最大缩放值阈值，默认10000")
    parser.add_argument("--flyers_num", type=int, default=25, 
                       help="移除飞点的最临近点数，默认25")
    parser.add_argument("--flyers_dis", type=float, default=10, 
                       help="移除飞点的距离因子，默认10")
    
    # 跳过选项
    parser.add_argument("--skip_clean", action="store_true", 
                       help="跳过clean步骤")
    
    args = parser.parse_args()
    
    # 参数处理
    input_dir = args.input
    output_dir = args.output
    enu_origin = (args.enu_origin[0], args.enu_origin[1]) if args.enu_origin else (0.0, 0.0)
    
    # 验证输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行LOD构建
    success = build_lod_from_ply(
        input_dir=input_dir,
        output_dir=output_dir,
        enu_origin=enu_origin,
        tile_zoom=args.tile_zoom,
        tile_resolution=args.tile_resolution,
        tile_error=args.tile_error,
        lod_factor=args.lod_factor,
        use_fine_lod=args.use_fine_lod,
        target_reduction_ratio=args.target_reduction_ratio,
        min_alpha=args.min_alpha,
        max_scale=args.max_scale,
        flyers_num=args.flyers_num,
        flyers_dis=args.flyers_dis,
        skip_clean=args.skip_clean
    )
    
    if success:
        print("\n🎉 LOD构建成功完成！")
        sys.exit(0)
    else:
        print("\n❌ LOD构建失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
