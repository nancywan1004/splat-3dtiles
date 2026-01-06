"""
GPU加速工具模块
支持使用faiss进行GPU加速的KDTree查询

作者：基于faiss库实现
"""

import numpy as np
from typing import Optional, List, Tuple
import os

# 尝试导入faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# 尝试导入scipy作为回退
try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    KDTree = None


class GPUIndexWrapper:
    """
    GPU加速的KDTree包装器
    使用faiss实现，如果GPU不可用则回退到CPU
    """
    
    def __init__(self, positions: np.ndarray, use_gpu: bool = True, gpu_id: int = 0):
        """
        初始化GPU索引
        
        Args:
            positions: 点位置数组，形状为 (n_points, 3)
            use_gpu: 是否使用GPU
            gpu_id: GPU设备ID
        """
        self.positions = positions.astype('float32')
        self.n_points = len(positions)
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.gpu_id = gpu_id
        self.index = None
        self.cpu_index = None  # CPU回退索引
        self.gpu_resource = None
        
        # 检查GPU可用性
        if self.use_gpu:
            try:
                # 检查是否有GPU
                gpu_resources = faiss.StandardGpuResources()
                self.gpu_resource = gpu_resources
                # 创建GPU索引
                self._build_gpu_index()
                print(f"✓ 使用GPU (ID: {gpu_id}) 进行加速")
            except Exception as e:
                print(f"⚠️  GPU不可用，回退到CPU: {e}")
                self.use_gpu = False
                self._build_cpu_index()
        else:
            self._build_cpu_index()
    
    def _build_gpu_index(self):
        """构建GPU索引"""
        # 创建L2距离的平面索引（适合小规模数据）
        # 对于大规模数据，可以考虑使用IndexIVFFlat或IndexHNSWFlat
        cpu_index = faiss.IndexFlatL2(3)  # 3维空间
        
        # 将索引移到GPU
        try:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resource, self.gpu_id, cpu_index)
        except Exception as e:
            print(f"⚠️  无法将索引移到GPU {self.gpu_id}，尝试使用默认GPU: {e}")
            try:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
                self.gpu_id = 0
            except Exception as e2:
                print(f"⚠️  所有GPU都不可用，回退到CPU: {e2}")
                self.use_gpu = False
                self._build_cpu_index()
                return
        
        # 添加数据到索引
        self.index.add(self.positions)
    
    def _build_cpu_index(self):
        """构建CPU索引（使用scipy或faiss CPU）"""
        if SCIPY_AVAILABLE:
            print("使用scipy.spatial.KDTree (CPU)")
            self.cpu_index = KDTree(self.positions)
        elif FAISS_AVAILABLE:
            print("使用faiss CPU索引")
            self.cpu_index = faiss.IndexFlatL2(3)
            self.cpu_index.add(self.positions)
        else:
            raise RuntimeError("既没有scipy也没有faiss，无法构建索引")
    
    def query_ball_point(self, point: np.ndarray, radius: float) -> List[int]:
        """
        查询半径内的所有点
        
        Args:
            point: 查询点，形状为 (3,)
            radius: 查询半径
            
        Returns:
            索引列表
        """
        if self.use_gpu and self.index is not None:
            return self._query_ball_point_gpu(point, radius)
        else:
            return self._query_ball_point_cpu(point, radius)
    
    def _query_ball_point_gpu(self, point: np.ndarray, radius: float) -> List[int]:
        """
        使用GPU查询半径内的点
        
        注意：faiss的IndexFlatL2只支持k-NN查询，不支持半径查询
        我们使用k-NN查询，然后过滤距离
        """
        point = point.astype('float32').reshape(1, 3)
        radius_sq = radius * radius
        
        # 估算最大可能返回的点数（基于半径）
        # 对于均匀分布的点，可以使用体积估算
        # 保守估算：使用所有点（实际会过滤）
        k = min(self.n_points, 10000)  # 限制最大查询数，避免内存问题
        
        # 执行k-NN查询
        distances, indices = self.index.search(point, k)
        
        # 过滤距离小于半径的点
        valid_mask = distances[0] <= radius_sq
        valid_indices = indices[0][valid_mask].tolist()
        
        # 如果查询返回的点数等于k，可能还有更多点
        # 扩大搜索范围
        if len(valid_indices) == k and k < self.n_points:
            # 逐步扩大搜索范围
            for new_k in [k * 2, k * 4, self.n_points]:
                new_k = min(new_k, self.n_points)
                distances, indices = self.index.search(point, new_k)
                valid_mask = distances[0] <= radius_sq
                valid_indices = indices[0][valid_mask].tolist()
                if len(valid_indices) < new_k:
                    break
        
        return valid_indices
    
    def _query_ball_point_cpu(self, point: np.ndarray, radius: float) -> List[int]:
        """使用CPU查询半径内的点"""
        if self.cpu_index is not None and SCIPY_AVAILABLE and isinstance(self.cpu_index, KDTree):
            # 使用scipy的KDTree
            return self.cpu_index.query_ball_point(point, radius).tolist()
        elif self.cpu_index is not None and FAISS_AVAILABLE:
            # 使用faiss CPU
            point = point.astype('float32').reshape(1, 3)
            radius_sq = radius * radius
            k = min(self.n_points, 10000)
            distances, indices = self.cpu_index.search(point, k)
            valid_mask = distances[0] <= radius_sq
            valid_indices = indices[0][valid_mask].tolist()
            
            # 如果需要，扩大搜索范围
            if len(valid_indices) == k and k < self.n_points:
                for new_k in [k * 2, k * 4, self.n_points]:
                    new_k = min(new_k, self.n_points)
                    distances, indices = self.cpu_index.search(point, new_k)
                    valid_mask = distances[0] <= radius_sq
                    valid_indices = indices[0][valid_mask].tolist()
                    if len(valid_indices) < new_k:
                        break
            
            return valid_indices
        else:
            raise RuntimeError("没有可用的CPU索引")
    
    def batch_query_ball_point(self, points: np.ndarray, radius: float) -> List[List[int]]:
        """
        批量查询多个点的半径邻域
        
        Args:
            points: 查询点数组，形状为 (n_queries, 3)
            radius: 查询半径
            
        Returns:
            每个查询点的索引列表的列表
        """
        if self.use_gpu and self.index is not None:
            return self._batch_query_ball_point_gpu(points, radius)
        else:
            # CPU批量查询
            results = []
            for point in points:
                results.append(self._query_ball_point_cpu(point, radius))
            return results
    
    def _batch_query_ball_point_gpu(self, points: np.ndarray, radius: float) -> List[List[int]]:
        """使用GPU批量查询"""
        points = points.astype('float32')
        radius_sq = radius * radius
        n_queries = len(points)
        
        # 估算k值
        k = min(self.n_points, 10000)
        
        # 批量查询
        distances, indices = self.index.search(points, k)
        
        # 对每个查询点过滤距离
        results = []
        for i in range(n_queries):
            valid_mask = distances[i] <= radius_sq
            valid_indices = indices[i][valid_mask].tolist()
            
            # 如果返回的点数等于k，可能需要扩大搜索范围
            if len(valid_indices) == k and k < self.n_points:
                # 单独查询这个点
                point = points[i:i+1]
                for new_k in [k * 2, k * 4, self.n_points]:
                    new_k = min(new_k, self.n_points)
                    dists, idxs = self.index.search(point, new_k)
                    valid_mask = dists[0] <= radius_sq
                    valid_indices = idxs[0][valid_mask].tolist()
                    if len(valid_indices) < new_k:
                        break
            
            results.append(valid_indices)
        
        return results
    
    def cleanup(self):
        """清理资源"""
        if self.index is not None:
            del self.index
        if self.gpu_resource is not None:
            del self.gpu_resource
        if self.cpu_index is not None:
            del self.cpu_index


def check_gpu_available() -> Tuple[bool, int]:
    """
    检查GPU是否可用
    
    Returns:
        (是否可用, GPU数量)
    """
    if not FAISS_AVAILABLE:
        return False, 0
    
    try:
        gpu_count = faiss.get_num_gpus()
        return gpu_count > 0, gpu_count
    except Exception:
        return False, 0


def get_default_gpu_id() -> int:
    """获取默认GPU ID"""
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    try:
        return int(gpu_id)
    except ValueError:
        return 0

