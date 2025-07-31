import torch
import torch.nn as nn
import dgl
import pprint
import itertools
from functools import reduce
import dgl.function as fn
from models.rho_encoder import RHOEncoder
from gnn_zoo.homogeneous_gnns import GCN
from utils.nn_modules import MLP, GraphStitchHead
from typing import Dict, List, Optional
from dgl import DGLGraph

class Uni_RHO_GAD_Predictor(nn.Module):
    """
    The main predictor for the Uni-RHO-GAD model.
    
    这个模型包含一个共享的预训练编码器、并行的RHOEncoder，以及并行的融合-预测头。
    """

    def __init__(self, 
                 pretrain_model, 
                 feature_adapter: nn.Module, 
                 embed_dims: int, 
                 num_classes: int,
                 is_single_graph: bool,
                 all_tasks: List[str] = ['n', 'e', 'g'], 
                 cross_modes: List[str] = ['ng2ng', 'g2ng', 'n2ng'],
                 base_gnn_layers: int = 2, 
                 final_mlp_layers: int = 2, 
                 gna_projection_dim: int = 128, 
                 dropout_rate: float = 0.5,
                 activation: str = 'ReLU',
                 residual: bool = True,
                 norm: str = 'layernorm'):
        super().__init__()

        self.is_single_graph = is_single_graph
        
        # 1. 共享的预训练模型 (用于初始特征提取)
        self.pretrain_model = pretrain_model
        # 确保预训练模型的输出维度与特征适配器的输入维度一致
        self.feature_adapter = feature_adapter

        self.all_tasks = all_tasks
        
        # 2. 并行的 RHO-Encoder 分支
        self.rho_encoders = nn.ModuleDict()
        for task in all_tasks:
            # 所有分支都使用GCN作为基础，因为它们都处理图结构
            base_gnn = GCN(
                in_dim=embed_dims, num_hidden=embed_dims, out_dim=embed_dims,
                num_layers=base_gnn_layers, dropout=dropout_rate,
                activation=activation,
                residual=residual,
                norm=norm,
                encoding=True
            )
            self.rho_encoders[task] = RHOEncoder(base_gnn, embed_dims, gna_projection_dim)

        # 3. 并行的融合-预测头 (GraphStitchHeads)
        self.fusion_heads = nn.ModuleDict({
            # 将 'ng2g' 转换为 'ng_to_g' 作为字典的键，提高可读性
            # 将原始的 'ng2g' 字符串传递给 GraphStitchHead
            mode.replace('2', '_to_'): GraphStitchHead(
                cross_mode=mode, 
                embed_dim=embed_dims, 
                num_classes=num_classes,
                mlp_layers=final_mlp_layers, 
                dropout_rate=dropout_rate, 
                activation=activation
            ) 
            for mode in cross_modes
        })

        # 4. 共享的单类损失中心
        self.centers = nn.ParameterDict({
            task: nn.Parameter(torch.randn(1, embed_dims)) for task in all_tasks
        })

    def forward(self, 
                batched_inputs: Dict[str, DGLGraph], 
                original_graph: Optional[DGLGraph] = None,
                batched_target_ids: Optional[Dict[str, torch.Tensor]] = None,
                normal_masks: Optional[Dict[str, torch.Tensor]] = None):

        # --- Step 1: 高效的初始嵌入提取 ---
        initial_embeddings = {}
        device = next(self.parameters()).device # 获取模型所在的设备

        if self.is_single_graph:
            if original_graph is None:
                raise ValueError("`original_graph` must be provided for single-graph scenarios.")
            
            original_graph = original_graph.to(device)
            with torch.no_grad():
                global_h = self.pretrain_model.to(device).embed(original_graph, original_graph.ndata['feature'])
            
            for task, g_batch in batched_inputs.items():
                if g_batch.num_nodes() > 0:
                    original_node_ids = g_batch.ndata[dgl.NID].to(device)
                    initial_embeddings[task] = global_h[original_node_ids]
        else:
            with torch.no_grad():
                for task, g_batch in batched_inputs.items():
                     if g_batch.num_nodes() > 0:
                        g_batch = g_batch.to(device)
                        initial_embeddings[task] = self.pretrain_model.to(device).embed(g_batch, g_batch.ndata['feature'])
        
        adapted_embeddings = {
            task: self.feature_adapter(h) for task, h in initial_embeddings.items()
        }

        # --- Step 2: 并行通过RHOEncoder，并进行Readout ---
        branch_representations = {}
        total_gna_loss, total_one_class_loss = 0.0, 0.0
        
        for task, h_adapted in adapted_embeddings.items():
            if task not in self.rho_encoders: continue

            g_batch = batched_inputs[task]
            h_robust_nodes, loss_gna = self.rho_encoders[task](g_batch, h_adapted)
            
            # Readout: 根据任务类型提取最终表示
            task_level_rep = None

            # 先将要池化的特征存入图中
            g_batch.ndata['h_for_readout'] = h_robust_nodes

            if task == 'n':
                if not self.is_single_graph:
                    # 对于多图节点任务，直接使用所有节点的表示
                    task_level_rep = h_robust_nodes
                else:
                    # 单图节点任务：对每个子图进行池化
                    task_level_rep = dgl.mean_nodes(g_batch, 'h_for_readout')
            elif task == 'g':
                # 图任务：对每个图进行池化
                task_level_rep = dgl.mean_nodes(g_batch, 'h_for_readout')
            elif task == 'e':
                # 边任务: 对端点子图池化
                task_level_rep = dgl.mean_nodes(g_batch, 'h_for_readout')

            # 清理临时特征
            g_batch.ndata.pop('h_for_readout')
            
            if task_level_rep is not None:
                branch_representations[task] = task_level_rep
            
             # --- Step 3: 累加损失 (仅在训练时) ---
            if self.training:
                if loss_gna is not None: 
                    total_gna_loss += loss_gna

                # 计算 one-class 损失
                if normal_masks and task in normal_masks and normal_masks[task].sum() > 0:
                    # one-class 损失作用于Readout后的任务级别表示
                    normal_reps = task_level_rep[normal_masks[task]]
                    center = self.centers[task]
                    one_class_loss = torch.pow(normal_reps - center, 2).mean()
                    total_one_class_loss += one_class_loss
        

        # --- Step 4: 通过并行的融合头进行预测 ---
        # 对于多图场景，节点和图任务都源自同一个批处理图
        graph_context = None
        if not self.is_single_graph and 'g' in batched_inputs:
            graph_context = batched_inputs['g']

        all_logits = {
            mode_name: head(branch_representations, 
                            is_single_graph=self.is_single_graph,
                            graph_for_node_task=graph_context) 
            for mode_name, head in self.fusion_heads.items()
        }
        
        # 过滤掉因缺少输入而返回None的结果
        all_logits = {k: v for k, v in all_logits.items() if v is not None}
        
        losses = {'gna': total_gna_loss, 'one_class': total_one_class_loss}

        return all_logits, losses


    def predict_anomaly_score(self, 
                              batched_inputs: dict, 
                              original_graph: Optional[DGLGraph] = None,
                              target_ids: Optional[Dict[str, torch.Tensor]] = None):
        """
            独立的推断方法，用于计算各任务的异常分数。
            异常分数定义为到one-class中心的距离。
        """
        self.eval() # 设置为评估模式
        with torch.no_grad():
            # --- Step 1: 嵌入提取 (与forward中逻辑相同) ---
            initial_embeddings = {}
            device = next(self.parameters()).device
            if self.is_single_graph:
                if original_graph is None: raise ValueError("`original_graph` must be provided.")
                global_h = self.pretrain_model.embed(original_graph, original_graph.ndata['feature'])
                for task, g_batch in batched_inputs.items():
                    if g_batch.num_nodes() > 0:
                        original_node_ids = g_batch.ndata[dgl.NID]
                        initial_embeddings[task] = global_h[original_node_ids]
            else:
                for task, g_batch in batched_inputs.items():
                    if g_batch.num_nodes() > 0:
                        initial_embeddings[task] = self.pretrain_model.embed(g_batch, g_batch.ndata['feature'])

            adapted_embeddings = {task: self.feature_adapter(h) for task, h in initial_embeddings.items()}
            
            # --- Step 2: 计算表示和异常分数 ---
            anomaly_scores = {}
            for task, h_adapted in adapted_embeddings.items():
                if task not in self.rho_encoders: continue
                
                g_batch = batched_inputs[task]
                h_robust_nodes, _ = self.rho_encoders[task](g_batch, h_adapted)
                
                # Readout
                task_level_rep = None

                g_batch.ndata['h_for_readout'] = h_robust_nodes

                if task == 'n': 
                    if not self.is_single_graph and batched_target_ids and task in batched_target_ids:
                        task_level_rep = h_robust_nodes[batched_target_ids[task]]
                    else:
                        task_level_rep = dgl.mean_nodes(g_batch, 'h_for_readout')
                elif task == 'g': 
                    task_level_rep = dgl.mean_nodes(g_batch, 'h_for_readout')
                elif task == 'e':
                    task_level_rep = dgl.mean_nodes(g_batch, 'h_for_readout')

                g_batch.ndata.pop('h_for_readout')
                
                # 计算到中心的距离作为异常分数
                if task_level_rep is not None:
                    center = self.centers[task]
                    distances = torch.sum((task_level_rep - center)**2, dim=1)
                    anomaly_scores[task] = distances
                    
        return anomaly_scores