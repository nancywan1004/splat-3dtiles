# GPU加速使用指南

## 概述

本项目现已支持GPU加速的LOD构建，使用faiss库实现GPU加速的KDTree查询，可以大幅提升大规模点云的处理速度。

## 安装依赖

### 1. 安装faiss-gpu

```bash
# 使用conda安装（推荐）
conda install -c conda-forge faiss-gpu

# 或使用pip安装（需要CUDA环境）
pip install faiss-gpu
```

### 2. 验证安装

```python
import faiss
print(f"GPU数量: {faiss.get_num_gpus()}")
```

## 使用方法

### 基本用法

在命令行中添加 `--use_gpu` 参数即可启用GPU加速：

```bash
python3 main.py \
  --input ./data/drone/split/tile_20_524288_524289.ply \
  --output ./data/drone/build_fine \
  --tile_zoom 20 \
  --target_reduction_ratio 0.8 \
  --tile_resolution 0.03 \
  --use_fine_lod \
  --use_gpu \
  --gpu_id 0
```

### 参数说明

- `--use_gpu`: 启用GPU加速（需要安装faiss-gpu）
- `--gpu_id`: 指定使用的GPU设备ID，默认为0

### 多GPU使用

如果有多个GPU，可以为不同的任务分配不同的GPU：

```bash
# GPU 0处理任务1
CUDA_VISIBLE_DEVICES=0 python3 main.py --input file1.ply ... --use_gpu --gpu_id 0

# GPU 1处理任务2
CUDA_VISIBLE_DEVICES=1 python3 main.py --input file2.ply ... --use_gpu --gpu_id 0
```

## 性能提升

### 预期加速效果

| 点数 | CPU时间 | GPU时间 | 加速比 |
|------|---------|---------|--------|
| 100万 | 5-10分钟 | 1-2分钟 | 3-5倍 |
| 1000万 | 40-75分钟 | 5-15分钟 | 5-10倍 |
| 3000万+ | 数小时 | 15-30分钟 | 10-50倍 |

**注意：** 实际加速效果取决于：
- GPU型号（高端GPU效果更好）
- 点云密度
- 距离阈值大小
- 数据传输开销

## 自动回退

如果GPU不可用或未安装faiss-gpu，代码会自动回退到CPU模式：

- 如果没有GPU，自动使用CPU
- 如果没有安装faiss-gpu，自动使用scipy.KDTree
- 程序会显示警告信息，但不会中断执行

## 注意事项

1. **内存要求**：GPU需要额外显存存储索引，建议至少8GB显存
2. **数据传输**：CPU↔GPU数据传输有开销，对于小规模点云（<100万点），可能CPU更快
3. **多进程**：GPU加速与多进程并行不冲突，每个进程可以独立使用GPU
4. **CUDA版本**：确保CUDA版本与faiss-gpu兼容

## 故障排除

### 问题1：找不到faiss模块

```
ImportError: No module named 'faiss'
```

**解决：** 安装faiss-gpu：
```bash
conda install -c conda-forge faiss-gpu
```

### 问题2：GPU不可用

```
⚠️  GPU不可用，回退到CPU
```

**检查：**
1. 确认安装了faiss-gpu（不是faiss-cpu）
2. 检查CUDA是否安装：`nvcc --version`
3. 检查GPU是否可见：`nvidia-smi`

### 问题3：显存不足

```
RuntimeError: CUDA out of memory
```

**解决：**
1. 减少batch size或点云规模
2. 使用CPU模式：移除 `--use_gpu` 参数
3. 关闭其他占用GPU的程序

## 技术细节

### 实现原理

1. **GPU索引构建**：使用faiss的IndexFlatL2构建L2距离索引
2. **半径查询**：由于faiss主要支持k-NN查询，我们使用k-NN+距离过滤实现半径查询
3. **批量查询**：支持批量查询多个点，提高GPU利用率

### 代码结构

- `gpu_utils.py`: GPU加速工具模块
  - `GPUIndexWrapper`: GPU索引包装器
  - `check_gpu_available()`: 检查GPU可用性
  - `get_default_gpu_id()`: 获取默认GPU ID

- `main_build_lod_tiles.py`: 标准LOD构建（已支持GPU）
- `main_build_fine_lod.py`: 精细LOD构建（已支持GPU）

## 示例

### 示例1：使用GPU加速精细LOD

```bash
python3 main.py \
  --input ./data/drone/split \
  --output ./data/drone/build_fine \
  --skip_split \
  --skip_clean \
  --tile_zoom 20 \
  --target_reduction_ratio 0.8 \
  --tile_resolution 0.03 \
  --use_fine_lod \
  --use_gpu \
  --lod_levels 1
```

### 示例2：多GPU并行处理

```bash
# 终端1：GPU 0
CUDA_VISIBLE_DEVICES=0 python3 main.py --input file1.ply --output out1 --use_gpu --gpu_id 0 &

# 终端2：GPU 1
CUDA_VISIBLE_DEVICES=1 python3 main.py --input file2.ply --output out2 --use_gpu --gpu_id 0 &

wait
```

## 总结

GPU加速可以显著提升大规模点云LOD构建的速度，特别是对于超过1000万点的点云。建议：

1. **大规模点云（>1000万点）**：强烈推荐使用GPU加速
2. **中等规模点云（100万-1000万点）**：根据硬件情况选择
3. **小规模点云（<100万点）**：CPU可能更快（避免数据传输开销）

