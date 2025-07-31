import torch
import dgl
import unittest
# 确保你的项目根目录在PYTHONPATH中
from models.rho_encoder import AdaFreqFilter

def get_test_laplacian(g):
    """
    辅助函数：从DGL图计算稀疏的PyTorch归一化拉普拉斯矩阵。
    (兼容性版本，不使用 g.adj() 来避免 dgl-sparse 依赖问题)
    """
    # 1. 添加自环
    g_with_loop = dgl.add_self_loop(g)
    n = g_with_loop.num_nodes()
    
    # 2. 获取边索引
    src, dst = g_with_loop.edges()
    
    # 3. 创建稀疏邻接矩阵 A
    adj_sparse = torch.sparse_coo_tensor(
        torch.stack([src, dst]),
        torch.ones(g_with_loop.num_edges()),
        (n, n)
    )
    
    # 4. 计算 D^-1/2
    deg = g_with_loop.in_degrees().float().clamp(min=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    
    # 5. 构建稀疏对角矩阵 D^-1/2
    D_inv_sqrt_sparse = torch.sparse_coo_tensor(
        torch.tensor([range(n), range(n)]),
        deg_inv_sqrt,
        (n, n)
    )
    
    # 6. 计算 D^-1/2 * A * D^-1/2
    norm_adj_sparse = torch.sparse.mm(D_inv_sqrt_sparse, torch.sparse.mm(adj_sparse, D_inv_sqrt_sparse))
    
    # 7. 计算 I - norm_adj
    I_sparse = torch.sparse_coo_tensor(torch.tensor([range(n), range(n)]), torch.ones(n), (n, n))
    
    return I_sparse - norm_adj_sparse

class TestAdaFreqFilter(unittest.TestCase):

    def setUp(self):
        """在每个测试前执行，准备测试数据"""
        self.num_nodes = 10
        self.embed_dim = 32
        # 创建一个随机图
        self.g = dgl.rand_graph(self.num_nodes, 20)
        self.features = torch.randn(self.num_nodes, self.embed_dim)
        self.laplacian = get_test_laplacian(self.g)
        self.filter = AdaFreqFilter(embed_dim=self.embed_dim)

    def test_output_shape(self):
        """测试输出形状是否正确"""
        # 测试跨通道视图
        h_ccr = self.filter(self.laplacian, self.features, view='cross_channel')
        self.assertEqual(h_ccr.shape, self.features.shape)

        # 测试逐通道视图
        h_cwr = self.filter(self.laplacian, self.features, view='channel_wise')
        self.assertEqual(h_cwr.shape, self.features.shape)
        print("AdaFreqFilter: Output shape test passed.")

    def test_parameter_learning(self):
        """测试参数是否可学习"""
        optimizer = torch.optim.SGD(self.filter.parameters(), lr=0.1)
        
        k_initial = self.filter.k_cross_channel.clone()
        K_initial = self.filter.K_channel_wise.clone()

        # 模拟一次训练迭代
        optimizer.zero_grad()
        
        # --- START: 修改这部分 ---
        # 让两个视图的输出都参与损失计算
        h_ccr = self.filter(self.laplacian, self.features, view='cross_channel')
        h_cwr = self.filter(self.laplacian, self.features, view='channel_wise')
        
        # 将两个输出相加作为虚拟损失
        loss = h_ccr.sum() + h_cwr.sum() 
        # --- END: 修改这部分 ---
        
        loss.backward()
        optimizer.step()

        # 检查两个参数是否都已更新
        self.assertFalse(torch.equal(self.filter.k_cross_channel, k_initial), "k_cross_channel was not updated!")
        self.assertFalse(torch.allclose(self.filter.K_channel_wise, K_initial), "K_channel_wise was not updated!")
        print("AdaFreqFilter: Parameter learning test passed.")

if __name__ == '__main__':
    unittest.main()