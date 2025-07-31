import os
import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling

NAME_MAP = {
    'n': "Node",
    'e': "Edge",
    'g': "Graph",
}

EPS = 1e-12 # for nan 防除零误差
ROOT_SEED = 3407

# 损失日志 格式化输出节点/边/图任务的损失值
def log_loss(tags:str, loss_item_dicts):
    for tag, loss_item_dict in zip(tags,loss_item_dicts):
        print(f"{tag} loss  ", end='')
        for k,v in loss_item_dict.items():
            print("  {}: {:.4f}".format(
                NAME_MAP[k],
                v
            ), end='')
        print("")


# ======================================================================
#   Model activation/normalization creation function
# ======================================================================
# 激活函数工厂
def obtain_act(name=None):
    """
    Return activation function module
    """

    if name is None:
        return nn.Identity()

    # --- START: 添加这一行 ---
    name_lower = name.lower()
    # --- END: 添加这一行 ---

    if name_lower == 'relu':
        act = nn.ReLU(inplace=True)
    elif name_lower == "gelu":
        act = nn.GELU()
    elif name_lower == "prelu":
        act = nn.PReLU()
    elif name_lower == "elu":
        act = nn.ELU()
    elif name_lower == "leakyrelu":
        act = nn.LeakyReLU()
    elif name_lower == "tanh":
        act = nn.Tanh()
    elif name_lower == "sigmoid":
        act = nn.Sigmoid()
    else:
        # 使用 f-string 格式化输出，更清晰
        raise NotImplementedError(f"Activation function '{name}' is not implemented.")

    return act

# 归一化层工厂
def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    elif name == "instancenorm":
        norm = partial(nn.InstanceNorm1d, affine=True, track_running_stats=True)
    else:
        return nn.Identity

    return norm

# 图池化器
def obtain_pooler(pooling):
    """
    Return pooling function module
    """
    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    return pooler

def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]

def set_seed(seed=ROOT_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.use_deterministic_algorithms(mode=True)

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)    # 余弦相似度的缩放

    loss = loss.mean()
    return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                     (0 for the negative class and 1 for the positive class).
            weight (torch.Tensor, optional): 每个类别的权重 (C,)。如果提供，它将覆盖初始化的alpha。
        """
        # 1. 计算交叉熵损失 (不进行reduction，以便逐样本加权)
        #    我们直接使用 PyTorch 内置的 cross_entropy，它在数值上更稳定
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 2. 计算 pt (预测正确类别的概率)
        #    首先获取每个样本的预测概率
        p = F.softmax(inputs, dim=1)
        #    然后根据真实标签，gather出预测正确类别的那个概率
        p_t = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze()

        # 3. 计算Focal Loss的调制因子 (modulating factor)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # 4. 应用类别权重 (alpha-balancing)
        if weight is not None:
            # 如果传入了动态权重，使用它
            # weight 是 (C,)，targets 是 (N,)，通过索引为每个样本分配权重
            alpha_t = weight.gather(dim=0, index=targets)
            loss = alpha_t * loss
        elif self.alpha is not None:
            # 如果没有传入动态权重，但初始化时设置了alpha，使用它
            # 注意：标准的alpha是用于正类的，所以需要构建一个(C,)的权重
            # 这里简化处理，假设alpha是正类的权重
            alpha_tensor = torch.tensor([1 - self.alpha, self.alpha], device=inputs.device)
            alpha_t = alpha_tensor.gather(dim=0, index=targets)
            loss = alpha_t * loss

        # 5. 应用 reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss