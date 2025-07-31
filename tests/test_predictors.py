import torch
import unittest
import dgl

# 添加路径
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from predictors.unirhogad_predictor import Uni_RHO_GAD_Predictor

class TestUniRHOGADPredictor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前运行一次，准备共享的资源"""
        print("\n--- Setting up TestUniRHOGADPredictor ---")
        cls.embed_dims = 32
        cls.num_classes = 2
        
        # 1. 实例化一个“全功能”模型
        cls.model = Uni_RHO_GAD_Predictor(
            in_feats=64, # 假设的预训练输入维度
            embed_dims=cls.embed_dims,
            num_classes=cls.num_classes,
            input_route=['n', 'e', 'g'],
            output_route=['n', 'e', 'g']
        )
        
        # 2. 构建一个结构化的批处理图
        # 创建一个包含2个小图的列表
        graph1 = dgl.rand_graph(10, 20) # 10个节点, 20条边
        graph2 = dgl.rand_graph(15, 30) # 15个节点, 30条边
        cls.graph_list = [graph1, graph2]
        cls.batched_graph = dgl.batch(cls.graph_list)
        
        # 记录关键尺寸
        cls.num_nodes_total = cls.batched_graph.num_nodes() # 10 + 15 = 25
        cls.num_edges_total = cls.batched_graph.num_edges() # 20 + 30 = 50
        cls.num_graphs_total = len(cls.graph_list) # 2
        
        # 3. 模拟所有级别的输入嵌入
        cls.precomputed_embeddings = {
            'n': torch.randn(cls.num_nodes_total, cls.embed_dims),
            'e': torch.randn(cls.num_edges_total, cls.embed_dims),
            'g': torch.randn(cls.num_graphs_total, cls.embed_dims)
        }
        
        # 4. 模拟正常样本掩码
        cls.normal_masks = {
            'n': torch.randint(0, 2, (cls.num_nodes_total,)).bool(),
            'e': torch.randint(0, 2, (cls.num_edges_total,)).bool(),
            'g': torch.randint(0, 2, (cls.num_graphs_total,)).bool()
        }
        
        # 5. 准备 g_info
        # 关键：将节点表示存入图中，以便 n->e 融合路径使用
        cls.batched_graph.ndata['h_stitch_source'] = cls.precomputed_embeddings['n']
        cls.g_info = {
            'n': cls.batched_graph,
            'e': cls.batched_graph,
            'g': cls.batched_graph
        }

    def test_training_forward_pass(self):
        """测试训练模式下的完整前向传播"""
        print("Testing forward pass in training mode...")
        self.model.train()
        
        logits, loss_one_class, loss_gna, scores = self.model(
            self.precomputed_embeddings, 
            self.normal_masks, 
            self.g_info
        )
        
        # 验证 Logits
        self.assertIn('n', logits)
        self.assertIn('e', logits)
        self.assertIn('g', logits)
        self.assertEqual(logits['n'].shape, (self.num_nodes_total, self.num_classes))
        self.assertEqual(logits['e'].shape, (self.num_edges_total, self.num_classes))
        self.assertEqual(logits['g'].shape, (self.num_graphs_total, self.num_classes))
        
        # 验证损失
        self.assertIsInstance(loss_one_class, torch.Tensor)
        self.assertTrue(torch.is_tensor(loss_one_class))
        self.assertIsInstance(loss_gna, torch.Tensor)
        self.assertTrue(torch.is_tensor(loss_gna))
        
        # 验证评估模式下的输出 (scores) 在训练模式下为空
        self.assertDictEqual(scores, {})
        
        print("Training forward pass test passed.")

    def test_evaluation_forward_pass(self):
        """测试评估模式下的完整前向传播"""
        print("Testing forward pass in evaluation mode...")
        self.model.eval()
        
        with torch.no_grad():
            logits, loss_one_class, loss_gna, scores = self.model(
                self.precomputed_embeddings, 
                self.normal_masks, 
                self.g_info
            )
        
        # 验证 Logits (与训练模式相同)
        self.assertEqual(logits['n'].shape, (self.num_nodes_total, self.num_classes))
        self.assertEqual(logits['e'].shape, (self.num_edges_total, self.num_classes))
        self.assertEqual(logits['g'].shape, (self.num_graphs_total, self.num_classes))
        
        # 验证损失 (在评估模式下应为0或特定值)
        self.assertEqual(loss_one_class, 0)
        self.assertIsNone(loss_gna)
        
        # 验证异常分数
        self.assertIn('n', scores)
        self.assertIn('e', scores)
        self.assertIn('g', scores)
        self.assertEqual(scores['n'].shape, (self.num_nodes_total,))
        self.assertEqual(scores['e'].shape, (self.num_edges_total,))
        self.assertEqual(scores['g'].shape, (self.num_graphs_total,))
        
        print("Evaluation forward pass test passed.")

if __name__ == '__main__':
    unittest.main()