# PLY空间切片工具 (PLY Spatial Slicer)

## 概述

PLY空间切片工具是一个专门用于处理PLY格式3D高斯点云文件的空间分布切片工具。它可以将大型PLY文件按照空间分布切分为指定数量的块，支持均匀分布和自适应分布两种模式。

## 主要特性

- ✅ **专门针对PLY格式**: 完全支持3D Gaussian Splatting的PLY文件格式
- ✅ **空间分布切片**: 基于3D空间位置进行智能切片，而非简单的点数分割
- ✅ **指定块数量**: 可以精确指定要切分的块数量
- ✅ **多种分布模式**: 支持均匀分布和自适应分布
- ✅ **批量处理**: 支持单文件和目录批量处理
- ✅ **详细信息输出**: 生成包含切片信息的JSON文件
- ✅ **保持PLY格式**: 输入和输出都是标准PLY格式
- ✅ **球谐系数保持**: 完整保持原始PLY文件的球谐系数信息

## 安装依赖

```bash
pip install plyfile numpy tqdm
```

## 使用方法

### 基本用法

```bash
# 将单个PLY文件切分为8个空间块
python ply_spatial_slicer.py -i input.ply -o output_blocks/ -n 8

# 将目录中的所有PLY文件切分为16个空间块
python ply_spatial_slicer.py -i input_dir/ -o output_blocks/ -n 16

# 使用自适应分布模式切分为27个块
python ply_spatial_slicer.py -i input.ply -o output_blocks/ -n 27 --mode adaptive
```

### 参数说明

- `-i, --input`: 输入PLY文件或包含PLY文件的目录
- `-o, --output`: 输出目录，切片后的文件将保存在此目录
- `-n, --num-blocks`: 目标块数量（必需参数）
- `--mode`: 分布模式
  - `uniform`: 均匀分布（默认）- 将空间均匀划分为网格
  - `adaptive`: 自适应分布 - 基于点密度进行智能划分

### 输出文件

工具会在输出目录中生成以下文件：

1. **切片文件**: `原文件名_block_XXXX.ply`
   - 每个空间块对应一个PLY文件
   - 保持原始PLY格式和所有属性

2. **信息文件**: `spatial_slice_info.json`
   - 包含切片的详细信息
   - 全局边界、块信息、统计数据等

## 分布模式详解

### 均匀分布 (Uniform)

- 将3D空间均匀划分为网格
- 自动计算最佳的3D网格分割（尽量接近立方体）
- 适用于点分布相对均匀的场景
- 处理速度快，内存占用低

### 自适应分布 (Adaptive)

- 基于点密度分布进行智能划分
- 密度高的区域会分配更多的块
- 适用于点分布不均匀的场景
- 处理时间稍长，但分布更合理

## 使用示例

### 示例1: 处理单个大型PLY文件

```bash
# 将一个大型PLY文件切分为8个空间块
python ply_spatial_slicer.py -i large_model.ply -o blocks/ -n 8

# 输出:
# blocks/large_model_block_0000.ply
# blocks/large_model_block_0001.ply
# ...
# blocks/large_model_block_0007.ply
# blocks/spatial_slice_info.json
```

### 示例2: 批量处理目录中的PLY文件

```bash
# 处理目录中的所有PLY文件，每个文件切分为16个块
python ply_spatial_slicer.py -i models/ -o output/ -n 16

# 如果models/目录包含: model1.ply, model2.ply
# 输出:
# output/model1_block_0000.ply ~ model1_block_0015.ply
# output/model2_block_0000.ply ~ model2_block_0015.ply
# output/spatial_slice_info.json
```

### 示例3: 使用自适应分布

```bash
# 对点分布不均匀的模型使用自适应分布
python ply_spatial_slicer.py -i uneven_model.ply -o adaptive_blocks/ -n 12 --mode adaptive
```

## 输出信息文件格式

`spatial_slice_info.json` 文件包含以下信息：

```json
{
  "metadata": {
    "version": "1.0",
    "generator": "PLY Spatial Slicer",
    "created": "2024-01-01 12:00:00",
    "input_files": ["model.ply"],
    "num_blocks": 8,
    "distribution_mode": "uniform",
    "total_points": 1000000
  },
  "global_bounds": {
    "min": [-10.0, -10.0, -5.0],
    "max": [10.0, 10.0, 5.0],
    "center": [0.0, 0.0, 0.0],
    "size": [20.0, 20.0, 10.0]
  },
  "blocks": {
    "0": {
      "block_id": 0,
      "bounds": [-10.0, -10.0, -5.0, 0.0, 0.0, 0.0],
      "center": [-5.0, -5.0, -2.5],
      "size": [10.0, 10.0, 5.0],
      "files": [
        {
          "filename": "model_block_0000.ply",
          "source_file": "model.ply",
          "point_count": 125000
        }
      ],
      "total_points": 125000
    }
  }
}
```

## 性能优化建议

1. **内存使用**: 对于超大型PLY文件，建议分批处理或增加系统内存
2. **块数量选择**: 
   - 块数量过多会产生很多小文件，管理复杂
   - 块数量过少可能无法充分利用空间分布优势
   - 建议根据数据规模选择8-64个块
3. **分布模式选择**:
   - 点分布均匀时使用uniform模式
   - 点分布不均匀时使用adaptive模式

## 与现有工具的区别

| 特性 | PLY Spatial Slicer | splat_slicer.py |
|------|-------------------|-----------------|
| 输入格式 | PLY专用 | SPLAT + PLY |
| 切片方式 | 空间分布 | 点数/网格 |
| 块数量控制 | 精确指定 | 估算 |
| 分布模式 | 均匀/自适应 | 固定网格 |
| 适用场景 | PLY格式专用处理 | 通用切片 |

## 故障排除

### 常见问题

1. **ImportError: No module named 'plyfile'**
   ```bash
   pip install plyfile
   ```

2. **内存不足错误**
   - 减少同时处理的文件数量
   - 增加系统内存
   - 使用更少的块数量

3. **输出块为空**
   - 检查输入PLY文件是否有效
   - 确认点坐标范围是否正常
   - 尝试减少块数量

4. **处理速度慢**
   - 使用uniform模式而非adaptive模式
   - 减少块数量
   - 确保有足够的磁盘空间

### 调试信息

运行时会显示详细信息：
- 扫描的文件数量和点数
- 全局空间边界
- 网格分割方案
- 处理进度
- 最终统计结果

## 技术实现

- **空间划分算法**: 基于3D包围盒的均匀网格划分
- **点分配策略**: 基于点坐标的空间包含判断
- **自适应分布**: 基于密度网格的智能划分（简化版）
- **文件格式**: 完全兼容标准PLY格式，保持所有原始属性

## 版本信息

- 版本: 1.0
- 兼容性: Python 3.6+
- 依赖: plyfile, numpy, tqdm

## 许可证

基于现有splat-3dtiles项目开发，遵循相同许可证。
