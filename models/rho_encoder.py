# 包含AdaFreq、GNA、RHOEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import scipy.sparse as sp
from gnn_zoo.homogeneous_gnns import GCN, GIN, BWGNN

class AdaFreqFilter(nn.Module):

    def __init__(self, embed_dim: int):
        """
        Parameters
        ----------
        embed_dim : int
            The dimension of the feature channels for the channel-wise view.
        """
        super().__init__()
        # 可学习参数 k (用于跨通道视图)
        self.k_cross_channel = nn.Parameter(torch.randn(1))
        
        # 可学习参数 K (用于逐通道视图)
        # 形状为 (1, embed_dim) 以便与 (N, embed_dim) 的特征矩阵进行广播
        self.K_channel_wise = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, L: torch.sparse.Tensor, H: torch.Tensor, view: str) -> torch.Tensor:

        if view == 'cross_channel':
            # 高效实现: H - k * (L @ H)
            # L是稀疏矩阵，H是稠密矩阵，使用torch.sparse.mm进行高效乘法
            return H - self.k_cross_channel * torch.sparse.mm(L, H)
        elif view == 'channel_wise':
            # 逐通道滤波: H - L @ (H * K)
            # H * self.K_channel_wise 利用了PyTorch的广播机制
            return H - torch.sparse.mm(L, H * self.K_channel_wise)
        else:
            raise ValueError(f"Invalid view: {view}. Must be 'cross_channel' or 'channel_wise'.")


class GNA(nn.Module):

    def __init__(self, embed_dim: int, projection_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.projection_head_view1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, projection_dim)
        )
        self.projection_head_view2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, projection_dim)
        )
        self.temperature = temperature

    def _sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def forward(self, h_view1: torch.Tensor, h_view2: torch.Tensor) -> torch.Tensor:
        z_view1 = self.projection_head_view1(h_view1)
        z_view2 = self.projection_head_view2(h_view2)
        
        sim_1_to_2 = self._sim(z_view1, z_view2)
        sim_2_to_1 = sim_1_to_2.t()
        
        labels = torch.arange(z_view1.size(0), device=z_view1.device)
        
        loss_1_to_2 = F.cross_entropy(sim_1_to_2 / self.temperature, labels)
        loss_2_to_1 = F.cross_entropy(sim_2_to_1 / self.temperature, labels)
        
        return (loss_1_to_2 + loss_2_to_1) / 2

class RHOEncoder(nn.Module):

    def __init__(self, base_gnn: nn.Module, embed_dim: int, gna_projection_dim: int = 128):
        super().__init__()
        self.base_gnn = base_gnn
        self.ada_freq_filter = AdaFreqFilter(embed_dim=embed_dim)
        self.gna_module = GNA(embed_dim, projection_dim=gna_projection_dim)

    def _get_laplacian(self, g: dgl.DGLGraph) -> torch.sparse.Tensor:

        # 1. 添加自环
        g_with_loop = dgl.add_self_loop(g)
        n = g_with_loop.num_nodes()
        
        # 2. 获取边索引
        src, dst = g_with_loop.edges()
        
        # 3. 创建稀疏邻接矩阵 A
        # 将设备与输入图的设备保持一致
        device = g.device
        adj_sparse = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            torch.ones(g_with_loop.num_edges(), device=device),
            (n, n)
        )
        
        # 4. 计算 D^-1/2
        deg = g_with_loop.in_degrees().float().clamp(min=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        
        # 5. 构建稀疏对角矩阵 D^-1/2
        D_inv_sqrt_sparse = torch.sparse_coo_tensor(
            torch.tensor([range(n), range(n)], device=device),
            deg_inv_sqrt,
            (n, n)
        )
        
        # 6. 计算 D^-1/2 * A * D^-1/2
        norm_adj_sparse = torch.sparse.mm(D_inv_sqrt_sparse, torch.sparse.mm(adj_sparse, D_inv_sqrt_sparse))
        
        # 7. 计算 I - norm_adj
        I_sparse = torch.sparse_coo_tensor(
            torch.tensor([range(n), range(n)], device=device),
            torch.ones(n, device=device),
            (n, n)
        )
        
        return I_sparse - norm_adj_sparse


    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):

        # 在执行任何操作前，确保图和特征在同一个设备上
        device = h.device
        g = g.to(device)

        # 1. 使用基础GNN提取初始嵌入
        # base_gnn (GCN, GIN, etc.) is expected to handle batched graphs.
        base_h = self.base_gnn(g, h)
        
        # 2. 计算拉普拉斯矩阵并进行自适应滤波
        L = self._get_laplacian(g) 
        h_ccr = self.ada_freq_filter(L, base_h, view='cross_channel')
        h_cwr = self.ada_freq_filter(L, base_h, view='channel_wise')

        # 3. 计算GNA损失 (仅在训练时)
        loss_gna = None
        if self.training:
            # GNA module also works on batched node features correctly.
            loss_gna = self.gna_module(h_ccr, h_cwr)
            
        # 4. 融合双视图表示作为最终输出
        # 简单地取平均值是一种有效且常用的融合策略
        final_h = (h_ccr + h_cwr) / 2.0
            
        return final_h, loss_gna