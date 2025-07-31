# 包含GraphMAE
from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl import DGLGraph

# 从我们的新位置导入
from gnn_zoo.homogeneous_gnns import GCN, GIN, BWGNN
from utils.misc import obtain_act, obtain_norm, sce_loss
from data.anomaly_generator import AnomalyGenerator

# 边掩码。伯努利采样生成掩码索引，用于图自监督学习中的边丢弃
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

# 边丢弃
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    
    E = graph.num_edges()   # 获取边的数量
    mask_rates = torch.FloatTensor(np.ones(E) * drop_rate)  # 生成掩码率
    masks = torch.bernoulli(1 - mask_rates) # 生成伯努利掩码
    edge_mask = masks.nonzero().squeeze(1)  # 获取掩码索引
    
    src, dst = graph.edges()
    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=graph.num_nodes())   # 重建子图
    ng = dgl.add_self_loop(ng)  # 添加自环保持连通性

    if return_edges:
        dsrc = src[~edge_mask]  # 获取未被掩码的源节点
        ddst = dst[~edge_mask]  # 获取未被掩码的目标节点
        return ng, (dsrc, ddst)
    return ng

# ======================================================================
#   Predictive SSL
# ======================================================================
# 基于掩码自编码器进行图结构预训练
class GraphMAE_PAA(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hid_dim: int,
            encoder_num_layer: int,
            decoder_num_layer: int,
            encoder_type: str = "gcn",
            decoder_type: str = "gcn",
            mask_ratio: float = 0.75,
            replace_ratio: float = 0.1,
            drop_edge_rate: float = 0.0,
            activation: str = "relu",
            norm: Optional[str] = "layernorm",
            residual: bool = True,
            loss_fn: str = "sce",
            alpha_l: float = 2.0,
            concat_hidden: bool = False,
            anomaly_generator: Optional[AnomalyGenerator] = None,
            w_recon: float = 0.5, w_contrastive: float = 1.0
         ):
        super(GraphMAE_PAA, self).__init__()
        self._mask_ratio = mask_ratio
        self._replace_ratio = replace_ratio
        self._mask_token_rate = 1.0 - self._replace_ratio
        self._drop_edge_rate = drop_edge_rate
        
        self.embed_dim = hid_dim # 预训练输出的嵌入维度
        encoder_out_dim = hid_dim
        decoder_in_dim = hid_dim

        self.anomaly_generator = anomaly_generator
        self.w_recon = w_recon
        self.w_contrastive = w_contrastive
        self.temperature = 0.1

        # 为对比损失创建一个投影头 (Projection Head)
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 128) # 投影到128维空间
        )

        # ======================================================================
        #   1. 编码器 (Encoder) - 支持多种GNN
        # ======================================================================
        print(f"Initializing GraphMAE_PAA Encoder: {encoder_type.upper()}")
        if encoder_type == "gcn":
            self.encoder = GCN(in_dim, hid_dim, hid_dim, encoder_num_layer, 0.0, activation, residual, norm, encoding=True)
        elif encoder_type == "gin":
            self.encoder = GIN(in_dim, hid_dim, hid_dim, encoder_num_layer, 0.0, activation, residual, norm, encoding=True)
        elif encoder_type == 'bwgnn':
            # BWGNN的输出维度是特殊的，需要单独处理
            self.encoder = BWGNN(in_dim=in_dim, num_hidden=hid_dim, encoding=True)
            # BWGNN的输出维度是 hid_dim * (d+1)，这里假设d=2
            encoder_out_dim = hid_dim * (self.encoder.d + 1)
            self.embed_dim = encoder_out_dim
            decoder_in_dim = encoder_out_dim # 解码器的输入维度也需要相应改变
        else:
            raise NotImplementedError(f"Encoder type '{encoder_type}' not supported.")

        # ======================================================================
        #   2. 解码器 (Decoder) - 结构更灵活
        # ======================================================================
        print(f"Initializing GraphMAE_PAA Decoder: {decoder_type.upper()}")
        if decoder_type == "gcn":
            self.decoder = GCN(decoder_in_dim, hid_dim, in_dim, decoder_num_layer, 0.0, activation, residual, norm, encoding=False)
        elif decoder_type == "mlp":
            # 使用一个简单的多层MLP作为解码器
            layers = [nn.Linear(decoder_in_dim, hid_dim), nn.ReLU()]
            for _ in range(decoder_num_layer - 2):
                layers.extend([nn.Linear(hid_dim, hid_dim), nn.ReLU()])
            layers.append(nn.Linear(hid_dim, in_dim))
            self.decoder = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"Decoder type '{decoder_type}' not supported.")

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # 编码器到解码器的投影层，处理可能的维度不匹配（如BWGNN）
        self.encoder_to_decoder = nn.Linear(encoder_out_dim, decoder_in_dim, bias=False)
        
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            return nn.MSELoss() # 均方误差损失
        elif loss_fn == "sce":
            return partial(sce_loss, alpha=alpha_l) # 交叉熵损失，带平滑参数
        else:
            raise NotImplementedError

    def info_nce_loss(self, z1, z2):
        """标准的 InfoNCE 对比损失"""
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # 正样本在对角线上
        labels = torch.arange(z1.size(0), device=z1.device)
        
        # 对称的损失
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss1 + loss2) / 2

    def embed(self, g, x):
        """用于在下游任务中获取节点嵌入的接口"""
        # 在推理时，我们使用原始的、完整的图
        with torch.no_grad():
            return self.encoder(g, x)

    # ======================================================================
    #   3. 混合掩码策略 (Mixed Masking)
    # ======================================================================
    def encoding_mask_noise(self, g, x):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        
        # a. 计算需要掩码的总节点数
        num_mask_nodes = int(self._mask_ratio * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        
        out_x = x.clone()
        
        if self._replace_ratio > 0 and num_mask_nodes > 0:
            # b. 计算两种掩码策略各自的节点数
            num_token_nodes = int(num_mask_nodes * self._mask_token_rate)
            num_noise_nodes = num_mask_nodes - num_token_nodes
            
            # c. 随机选择要被 [MASK] 替换的节点和要被噪声替换的节点
            mask_perm = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[mask_perm[:num_token_nodes]]
            noise_nodes = mask_nodes[mask_perm[num_token_nodes:]]

            # 确保有节点需要被替换
            if noise_nodes.numel() > 0:
                # d. 随机选择噪声节点的替换特征
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:noise_nodes.size(0)]
                # e. 将噪声节点替换为随机选择的特征
                out_x[noise_nodes] = x[noise_to_be_chosen]
            
            # f. 将掩码节点替换为 [MASK] 令牌
            if token_nodes.numel() > 0:
                # 我们显式地将 self.enc_mask_token 扩展到目标形状
                # .expand() 是一个高效的操作，它不会实际复制内存
                expanded_mask_token = self.enc_mask_token.expand(token_nodes.size(0), -1)
                # 替换掩码节点的特征为 [MASK] 令牌
                out_x[token_nodes] = expanded_mask_token
            
        elif num_mask_nodes > 0: # 如果不使用混合策略
            # 即使在非混合模式下，也使用显式扩展，保持代码健壮性
            expanded_mask_token = self.enc_mask_token.expand(num_mask_nodes, -1)
            out_x[mask_nodes] = expanded_mask_token
            
        return out_x, mask_nodes

    def contrastive_loss(self, z_normal, z_augmented_anomaly):
        """计算正常表示和其对应的人工异常表示之间的对比损失"""
        z_normal = F.normalize(z_normal, p=2, dim=1)
        z_augmented_anomaly = F.normalize(z_augmented_anomaly, p=2, dim=1)
        
        # 使用简单的余弦相似度损失，目标是让它们不相似（即点积接近-1）
        # 或者使用更复杂的 InfoNCE 损失
        # 简单版：最大化距离，即最小化相似度
        loss = (z_normal * z_augmented_anomaly).sum(dim=1).mean()
        return loss

    # def forward(self, g, x):
    #     # 1. 创建带噪声的图和特征
    #     # 边丢弃作为一种数据增强
    #     use_g = drop_edge(g, self._drop_edge_rate)
    #     # 应用混合掩码策略
    #     enc_x, mask_nodes = self.encoding_mask_noise(g, x)

    #     # 2. 编码
    #     enc_rep = self.encoder(use_g, enc_x)

    #     # 3. 解码
    #     # 投影到解码器的维度空间
    #     rep = self.encoder_to_decoder(enc_rep)
        
    #     # 如果解码器是GNN类型，需要再次掩码以防止信息泄露
    #     if isinstance(self.decoder, (GCN, GIN, BWGNN)):
    #         rep[mask_nodes] = 0
        
    #     # 根据解码器类型进行重建
    #     if isinstance(self.decoder, nn.Sequential): # MLP decoder
    #         recon = self.decoder(rep)
    #     else: # GNN decoder
    #         recon = self.decoder(use_g, rep)

    #     # 4. 计算损失 (仅在掩码节点上)
    #     loss = self.criterion(recon[mask_nodes], x[mask_nodes])
    #     return loss

    def forward(self, g, x):
        # --- 1. 创建两个不同的“视图” ---
        
        # 视图一：标准的 GraphMAE_PAA 增强 (边丢弃 + 节点掩码)
        g1 = drop_edge(g, self._drop_edge_rate)
        x1, mask_nodes = self.encoding_mask_noise(g, x)

        # 视图二：我们的人工异常增强
        # a. 选择一批节点进行增强
        num_to_aug = int(g.num_nodes() * self._mask_ratio) # 复用 mask_ratio
        nodes_to_aug = torch.randperm(g.num_nodes(), device=g.device)[:num_to_aug]
        
        # b. 生成人工异常
        aug_node_ids, perturbed_feats, perturbed_graph = self.anomaly_generator.generate_for_nodes(nodes_to_aug)
        
        # 如果未能生成异常，则只计算重建损失
        if aug_node_ids is None:
            rep1 = self.encoder(g1, x1)
            recon = self.decoder(g1, self.encoder_to_decoder(rep1))
            return self.criterion(recon[mask_nodes], x[mask_nodes])

        # --- 2. 通过同一个编码器，计算两个视图的表示 ---
        
        # 视图一的表示
        rep1 = self.encoder(g1, x1)
        
        # 视图二的表示
        full_perturbed_features = x.clone()
        perturbed_map = {nid.item(): feat for nid, feat in zip(aug_node_ids, perturbed_feats)}
        for i, feat in perturbed_map.items():
            full_perturbed_features[i] = feat
        
        device = x.device
        # 将从 AnomalyGenerator 返回的 CPU 图，移动到 GPU
        perturbed_graph = perturbed_graph.to(device)
        rep2_full = self.encoder(perturbed_graph, full_perturbed_features)
        
        # --- 3. 计算损失 ---
        
        # a. 对比损失 (主要目标)
        # 我们只对那些被增强的节点计算对比损失
        z1 = self.projection_head(rep1[aug_node_ids])
        z2 = self.projection_head(rep2_full[aug_node_ids])
        loss_cont = self.info_nce_loss(z1, z2)
        
        # b. 重建损失 (辅助目标)
        # 仍然在视图一上进行
        recon = self.decoder(g1, self.encoder_to_decoder(rep1))
        loss_recon = self.criterion(recon[mask_nodes], x[mask_nodes])
        
        # --- 4. 组合总损失 ---
        total_loss = self.w_contrastive * loss_cont + self.w_recon * loss_recon
        
        return total_loss



