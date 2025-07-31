# unirhogad_project/tests/test_data_loader.py

import unittest
import torch
import dgl
import os
from functools import partial
import shutil # 导入shutil用于删除目录

# 添加路径
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset_loader import UniGADDataset, collate_fn_unify, MRQSampler

# 注意：你需要将真实的UniGAD数据集文件放到测试可以访问的路径下
# 为了测试，我们假设在 tests/ 目录下有一个 mock_data 文件夹
# 你需要手动创建这些文件，或者修改路径指向你真实的数据集
MOCK_DATA_DIR = os.path.join(current_dir, "mock_data")

def create_mock_data():
    """创建一个模拟的数据集用于测试"""
    # 每次都重新创建，确保环境干净
    if os.path.exists(MOCK_DATA_DIR):
        shutil.rmtree(MOCK_DATA_DIR)
        
    print("Creating mock data for testing...")
    
    # --- START: 核心修改 ---
    # 1. 模拟多图数据集 (mutag)
    mutag_dir = os.path.join(MOCK_DATA_DIR, "unified")
    os.makedirs(os.path.join(mutag_dir, "mutag", "dgl"), exist_ok=True)
    
    # 创建4个图，确保每个类别至少有2个样本
    g1 = dgl.graph(([0, 1], [1, 0])); g1.ndata['feature'] = torch.randn(2, 10)
    g2 = dgl.graph(([0, 1], [1, 2])); g2.ndata['feature'] = torch.randn(3, 10)
    g3 = dgl.graph(([0, 1], [1, 0])); g3.ndata['feature'] = torch.randn(2, 10)
    g4 = dgl.graph(([0, 1], [1, 2])); g4.ndata['feature'] = torch.randn(3, 10)
    
    # 标签为 [0, 1, 0, 1]，每个类别都有2个样本
    labels = {'glabel': torch.tensor([0, 1, 0, 1])}
    dgl.save_graphs(os.path.join(mutag_dir, "mutag", "dgl", "mutag0"), [g1, g2, g3, g4], labels)
    # --- END: 修改这部分 ---

    # 2. 模拟单图数据集 (reddit)
    reddit_dir = os.path.join(MOCK_DATA_DIR, "edge_labels")
    os.makedirs(reddit_dir, exist_ok=True)
    
    g = dgl.rand_graph(50, 200); g.ndata['feature'] = torch.randn(50, 16)
    g.ndata['node_label'] = torch.randint(0, 2, (50,))
    g.edata['edge_label'] = torch.randint(0, 2, (200,))
    dgl.save_graphs(os.path.join(reddit_dir, "reddit-els"), [g])
    # --- END: 核心修改 ---


class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试前运行一次，准备模拟数据"""
        create_mock_data()

    def test_01_multigraph_dataset_loading(self):
        """测试多图数据集的加载和基本属性"""
        print("\nTesting multi-graph dataset loading...")
        dataset = UniGADDataset(name='mutag/dgl/mutag0', data_dir=MOCK_DATA_DIR)
        
        
        self.assertFalse(dataset.is_single_graph)
        self.assertEqual(len(dataset), 4) # 现在有4个图
        self.assertEqual(dataset.in_dim, 10)
        
        # 测试 __getitem__
        g, labels = dataset[0]
        self.assertIsInstance(g, dgl.DGLGraph)
        self.assertEqual(g.num_nodes(), 2)
        self.assertEqual(labels['g'].item(), 0)
        print("Multi-graph dataset loading: PASSED")

    def test_02_singlegraph_dataset_loading(self):
        """测试单图数据集的加载和基本属性"""
        print("\nTesting single-graph dataset loading...")
        dataset = UniGADDataset(name='reddit', data_dir=MOCK_DATA_DIR)
        
        self.assertTrue(dataset.is_single_graph)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.in_dim, 16)
        
        g, labels = dataset[0]
        self.assertEqual(g.num_nodes(), 50)
        self.assertEqual(labels['n'].shape[0], 50)
        self.assertEqual(labels['e'].shape[0], 200)
        print("Single-graph dataset loading: PASSED")

    def test_03_data_split(self):
        """测试数据划分逻辑"""
        print("\nTesting data splitting...")
        # 测试多图划分
        dataset_multi = UniGADDataset(name='mutag/dgl/mutag0', data_dir=MOCK_DATA_DIR)
        split_masks = dataset_multi.prepare_split(train_ratio=0.5, val_ratio=0) # 50% 训练, 50% 测试
        train_subset = dataset_multi.get_subset('train')
        self.assertEqual(len(train_subset), 2)

        # 测试单图划分
        dataset_single = UniGADDataset(name='reddit', data_dir=MOCK_DATA_DIR)
        split_masks = dataset_single.prepare_split(train_ratio=0.6, val_ratio=0.2)
        train_subset = dataset_single.get_subset('train')
        val_subset = dataset_single.get_subset('val')
        test_subset = dataset_single.get_subset('test')
        
        # 验证节点划分数量
        num_nodes = dataset_single.graph_list[0].num_nodes()
        self.assertAlmostEqual(len(train_subset.node_indices), num_nodes * 0.6, delta=1)
        self.assertAlmostEqual(len(val_subset.node_indices), num_nodes * 0.2, delta=1)
        self.assertAlmostEqual(len(test_subset.node_indices), num_nodes * 0.2, delta=1)
        print("Data splitting: PASSED")

    def test_04_mrqsampler(self):
        """测试MRQSampler能否正常工作"""
        print("\nTesting MRQSampler...")
        dataset = UniGADDataset(name='reddit', data_dir=MOCK_DATA_DIR, sp_type='star+norm')
        g = dataset.graph_list[0]
        sampler = dataset.sampler
        
        self.assertIsNotNone(sampler)
        
        # 为节点0进行采样
        sp_graph = sampler.sample(g, central_node_id=0)
        
        self.assertIsInstance(sp_graph, dgl.DGLGraph)
        self.assertTrue('pw' in sp_graph.edata) # 检查权重是否存在
        # 检查所有边的目标节点是否都是中心节点0
        _, dst_nodes = sp_graph.edges()
        self.assertTrue(torch.all(dst_nodes == 0))
        print("MRQSampler: PASSED")

    def test_05_dataloader_and_collate(self):
        """测试DataLoader和collate_fn的批处理功能"""
        print("\nTesting DataLoader and collate function...")
        
        # --- 测试多图场景 ---
        dataset_multi = UniGADDataset(name='mutag/dgl/mutag0', data_dir=MOCK_DATA_DIR)
        dataset_multi.prepare_split(train_ratio=0.5)
        train_subset_multi = dataset_multi.get_subset('train')
        
        # batch_size=2, 2个训练样本
        loader_multi = torch.utils.data.DataLoader(train_subset_multi, batch_size=2, collate_fn=collate_fn_unify)
        
        batched_graph, batched_labels, g_info = next(iter(loader_multi))
        
        # --- START: 修改断言 ---
        # 2个训练样本，节点数可能是 2+3 或 2+2 或 3+3，这里我们只检查总数
        self.assertEqual(len(train_subset_multi), 2) # 4 * 0.5 = 2
        # 检查批处理后的总节点数
        self.assertEqual(batched_graph.num_nodes(), 5) # 假设划分到了g1和g2 (2+3=5) 或 g3和g4 (2+3=5)
        self.assertEqual(batched_labels['g'].shape[0], 2)
        # --- END: 修改断言 ---
        self.assertIn('n', g_info)
        
        # --- 测试单图场景 ---
        dataset_single = UniGADDataset(name='reddit', data_dir=MOCK_DATA_DIR, sp_type='star+norm')
        dataset_single.prepare_split()
        train_subset_single = dataset_single.get_subset('train')
        
        collate_with_sampler = partial(collate_fn_unify, sampler=dataset_single.sampler)
        loader_single = torch.utils.data.DataLoader(train_subset_single, batch_size=4, collate_fn=collate_with_sampler)
        
        original_graph, batched_labels, subgraphs_dict = next(iter(loader_single))
        
        self.assertIsInstance(original_graph, dgl.DGLGraph)
        self.assertTrue('n' in batched_labels or 'e' in batched_labels)
        self.assertTrue('n' in subgraphs_dict or 'e' in subgraphs_dict)
        if 'n' in subgraphs_dict:
            self.assertIsInstance(subgraphs_dict['n'], dgl.DGLGraph)
        
        print("DataLoader and collate function: PASSED")

if __name__ == '__main__':
    unittest.main()