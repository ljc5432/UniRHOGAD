import torch
import dgl
import torch.nn as nn
from itertools import product
from functools import reduce

# ======================================================================
#   MLP: 通用多层感知机
# ======================================================================
class MLP(nn.Module):
    """
    一个通用的多层感知机模块。
    """
    def __init__(self, in_feats, h_feats, out_feats, num_layers=2, dropout_rate=0.5, activation='ReLU', output_activation=False):
        """
        Args:
            in_feats (int): 输入特征维度。
            h_feats (int): 隐藏层维度。
            out_feats (int): 输出特征维度。
            num_layers (int): 总层数 (包括输入和输出层)。
            dropout_rate (float): Dropout 比例。
            activation (str): 隐藏层激活函数。
            output_activation (bool): 是否在输出层后应用激活函数。
        """
        super().__init__()
        self.layers = nn.ModuleList()
        try:
            act_fn = getattr(nn, activation)()
        except AttributeError:
            print(f"Activation function '{activation}' not found in torch.nn, defaulting to ReLU.")
            act_fn = nn.ReLU()
        
        if num_layers == 1:
            # 如果只有一层，直接从输入映射到输出
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            # 输入层
            self.layers.append(nn.Linear(in_feats, h_feats))
            self.layers.append(act_fn)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            
            # 隐藏层
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(h_feats, h_feats))
                self.layers.append(act_fn)
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
            
            # 输出层
            self.layers.append(nn.Linear(h_feats, out_feats))

        if output_activation:
            self.layers.append(act_fn)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        纯粹的张量到张量的前向传播。
        
        Args:
            h (torch.Tensor): 输入的特征张量。
        
        Returns:
            torch.Tensor: 输出的特征张量。
        """
        for layer in self.layers:
            h = layer(h)
        return h
    

# ======================================================================
#   GraphStitchHead: 融合-预测头
# ======================================================================
class GraphStitchHead(nn.Module):
    """
    一个独立的模块，负责一种特定的 cross_mode 的信息融合和最终预测。
    它接收所有分支的图级别表示，并根据自己的路由策略进行融合和预测。
    """
    def __init__(self, cross_mode: str, embed_dim: int, num_classes: int, mlp_layers: int = 2, dropout_rate: float = 0.5, activation: str = 'ReLU'):
        """
        Args:
            cross_mode (str): 定义融合策略, e.g., "ng2ng".
            embed_dim (int): 输入的嵌入维度。
            num_classes (int): 最终分类器的输出类别数。
            **mlp_kwargs: 传递给内部MLP分类器的参数 (num_layers, dropout_rate, etc.)
        """
        super().__init__()
        self.cross_mode = cross_mode
        try:
            input_route_str, output_route_str = cross_mode.split('2')
            self.input_route = list(input_route_str)
            self.output_route = list(output_route_str)
        except ValueError:
            raise ValueError(f"Invalid cross_mode format: '{cross_mode}'. Expected format is 'source2target', e.g., 'ng2ng'.")
        
        # 1. GraphStitch 融合权重
        self.stitch_weights = nn.ParameterDict({
            f"{o_task}_from_{i_task}": nn.Parameter(torch.randn(1))
            for o_task in self.output_route for i_task in self.input_route
        })
        
        # 2. 最终的 MLP 预测器
        self.predictors = nn.ModuleDict({
            task: MLP(embed_dim, embed_dim, num_classes, 
                      num_layers=mlp_layers, 
                      dropout_rate=dropout_rate, 
                      activation=activation)
            for task in self.output_route
        })

    def forward(self, branch_representations: dict, is_single_graph: bool = False,
                 graph_for_node_task: dgl.DGLGraph=None) -> dict:

        logits_dict = {}


        if is_single_graph:
            # --- 单图场景：独立预测，不进行跨任务融合 ---
            for task, rep in branch_representations.items():
                if task in self.predictors:

                    logits_dict[task] = self.predictors[task](rep)

        else:
            # --- 多图场景：执行带有广播的GraphStitch融合逻辑 ---
            for o_task in self.output_route:
                if o_task not in self.predictors: continue

                # 1. 收集所有有效的输入源表示
                # inputs_to_fuse 是一个列表，存放加权后的表示张量
                inputs_to_fuse = []
                for i_task in self.input_route:
                    if i_task in branch_representations:
                        # 获取输入源的表示
                        source_rep = branch_representations[i_task]
                        
                        # 获取融合权重
                        weight = self.stitch_weights[f"{o_task}_from_{i_task}"]
                        
                        # 检查是否需要维度对齐
                        # 目标表示的形状，如果输出任务也在输入中，就用它，否则用源的形状（说明是n->n或g->g）
                        target_shape_ref = branch_representations.get(o_task, source_rep)
                        if source_rep.shape[0] != target_shape_ref.shape[0]:
                            # 维度不匹配，通常是 n <-> g 的情况
                            if o_task == 'n' and i_task == 'g': # g -> n, 需要广播
                                if graph_for_node_task is None: raise ValueError("Graph context needed for g->n fusion.")
                                aligned_rep = dgl.broadcast_nodes(graph_for_node_task, source_rep)
                            elif o_task == 'g' and i_task == 'n': # n -> g, 需要池化
                                if graph_for_node_task is None: raise ValueError("Graph context needed for n->g fusion.")
                                temp_feat_name = f"_temp_feat_{i_task}_to_{o_task}"
                                graph_for_node_task.ndata[temp_feat_name] = source_rep
                                aligned_rep = dgl.mean_nodes(graph_for_node_task, feat=temp_feat_name)
                                del graph_for_node_task.ndata[temp_feat_name] # 清理
                            else:
                                # 其他不匹配情况暂不处理
                                continue
                        else:
                            # 维度匹配，直接使用
                            aligned_rep = source_rep
                            
                        inputs_to_fuse.append(weight * aligned_rep)

                # 2. 如果没有任何有效的输入源，则跳过此任务
                if not inputs_to_fuse:
                    continue

                # 3. 融合所有加权后的输入表示
                fused_rep = reduce(torch.add, inputs_to_fuse)
                
                # 4. 可选的残差连接：如果输出任务本身也是一个输入源，可以添加残差
                if o_task in self.input_route and o_task in branch_representations:
                    fused_rep = fused_rep + branch_representations[o_task]

                # 5. 送入预测器
                logits_dict[o_task] = self.predictors[o_task](fused_rep)
        

        return logits_dict