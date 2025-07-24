# PLY格式支持说明

## 概述

本项目现已支持PLY格式的3D Gaussian Splatting文件切片功能。您可以直接使用PLY文件作为输入，无需预先转换为SPLAT格式。

## 新增功能

### 支持的文件格式
- ✅ `.splat` 文件 (原有支持)
- ✅ `.ply` 文件 (新增支持)
- ✅ 混合格式处理 (可同时处理SPLAT和PLY文件)

### 自动格式检测
程序会根据文件扩展名自动检测文件格式，无需手动指定。

### 格式保持 (新特性)
- ✅ **Split阶段格式保持**: 输入PLY文件，输出也是PLY格式的瓦片文件
- ✅ **Build LOD阶段格式保持**: LOD构建过程中保持原始文件格式
- ✅ **Clean阶段格式保持**: 数据清洗过程中保持文件格式一致性
- ✅ **端到端格式一致性**: 从输入到最终3D Tiles转换，中间过程保持格式统一

## 安装依赖

使用PLY格式需要安装额外的依赖库：

```bash
pip install plyfile
```

## 使用方法

### 基本用法

```bash
# 处理PLY文件
python main.py --input ./data/ply_folder --output ./output --enu_origin 118.91 32.12

# 处理混合格式文件夹 (包含.splat和.ply文件)
python main.py --input ./data/mixed_folder --output ./output --enu_origin 118.91 32.12
```

### 参数说明

所有原有参数保持不变：

- `--input`: 输入文件夹路径 (可包含.splat和.ply文件)
- `--output`: 输出3D Tiles文件夹路径
- `--enu_origin`: ENU坐标系原点经纬度
- `--tile_zoom`: 分块等级 (默认20)
- `--tile_resolution`: LOD精度参数 (默认0.1米)
- `--tile_error`: 几何误差参数 (默认1米)
- `--min_alpha`: 最小透明度阈值 (默认1.0)
- `--max_scale`: 最大缩放值阈值 (默认10000)
- `--flyers_num`: 移除飞点的邻近点数 (默认25)
- `--flyers_dis`: 移除飞点的距离因子 (默认10)

## PLY文件格式要求

### 必需的属性
PLY文件必须包含以下属性：

- `x`, `y`, `z`: 3D位置坐标
- `scale_0`, `scale_1`, `scale_2`: 高斯椭球的缩放参数
- `rot_0`, `rot_1`, `rot_2`, `rot_3`: 四元数旋转参数
- `opacity`: 透明度参数
- `f_dc_0`, `f_dc_1`, `f_dc_2`: 球谐系数的DC分量 (RGB颜色)

### 可选的属性 (3阶球谐系数) ✨新增✨
- `f_rest_0` 到 `f_rest_44`: 高阶球谐系数 (45个)
  - **1阶球谐**: `f_rest_0` 到 `f_rest_8` (9个系数)
  - **2阶球谐**: `f_rest_9` 到 `f_rest_23` (15个系数)
  - **3阶球谐**: `f_rest_24` 到 `f_rest_44` (21个系数)

### 球谐系数说明
- **总计48个系数**: 3个DC分量 + 45个高阶分量
- **视角相关颜色**: 高阶系数提供视角相关的颜色变化
- **向后兼容**: 如果文件中没有高阶系数，会自动填充为0
- **完整支持**: 支持标准3D Gaussian Splatting训练输出的完整PLY格式

### 数据格式说明
- `scale_*`: 存储为对数值，程序会自动应用exp()变换
- `opacity`: 存储为logit值，程序会自动应用sigmoid变换
- `f_dc_*`: 球谐系数，程序会自动转换为RGB颜色值
- `rot_*`: 四元数，程序会自动归一化并转换为内部格式

## 测试功能

### 基础PLY支持测试
运行测试脚本验证PLY支持：

```bash
python test_ply_support.py
```

测试脚本会：
1. 检查plyfile库是否已安装
2. 查找并测试PLY文件读取
3. 验证混合格式支持
4. 显示第一个点的详细信息

### 格式保持功能测试
运行格式保持测试脚本：

```bash
python test_format_preservation.py
```

测试脚本会：
1. 创建测试用的PLY和SPLAT文件
2. 验证读写一致性
3. 测试格式自动检测
4. 确认格式保持功能正常工作

### 3阶球谐系数测试 ✨新增✨
运行球谐系数测试脚本：

```bash
python test_spherical_harmonics.py
```

测试脚本会：
1. 创建包含完整3阶球谐系数的测试PLY文件
2. 验证48个球谐系数的读写
3. 测试向后兼容性 (只有DC分量的文件)
4. 确认球谐系数精度保持

## 性能说明

- PLY文件读取速度略慢于SPLAT文件 (需要解析文本/二进制格式)
- PLY文件写入需要额外的格式转换开销
- 切片性能与SPLAT文件基本相同
- 支持多进程并行处理，可同时处理多个PLY文件
- 格式保持功能不会显著影响处理速度

## 兼容性

- 兼容标准3D Gaussian Splatting训练输出的PLY文件
- 支持二进制和ASCII格式的PLY文件
- 与现有SPLAT格式完全兼容，可混合使用

## 故障排除

### 常见问题

1. **ImportError: No module named 'plyfile'**
   ```bash
   pip install plyfile
   ```

2. **PLY文件读取失败**
   - 检查PLY文件是否包含必需的属性
   - 确认文件格式正确 (标准3DGS输出)

3. **内存不足**
   - 对于大型PLY文件，确保有足够的内存
   - 考虑分批处理大文件

### 调试信息

程序会显示详细的处理信息：
- 找到的文件数量和格式分布
- 每个文件的点数量
- 处理进度和性能统计

## 示例

```bash
# 完整示例
python main.py \
  --input ./data/gaussian_models \
  --output ./output/3dtiles \
  --enu_origin 118.91083364082562 32.116922266350315 \
  --tile_zoom 20 \
  --tile_resolution 0.1 \
  --tile_error 1.0 \
  --min_alpha 0.1 \
  --max_scale 1000
```

处理完成后，输出目录将包含标准的Cesium 3D Tiles格式文件，可直接用于Web渲染。
