# 预留 未来实现RGCN、HGT等
# gnn_zoo/heterogeneous_gnns.py

# This file is reserved for future implementation of GNNs that operate on
# heterogeneous graphs, such as RGCN (Relational GCN) or HGT (Heterogeneous
# Graph Transformer).

# Example class structure:
#
# import dgl.nn.pytorch as dglnn
#
# class RGCN(nn.Module):
#     def __init__(self, ...):
#         super().__init__()
#         # ... implementation ...
#
#     def forward(self, g, features_dict):
#         # ... implementation ...
#         return output_features_dict