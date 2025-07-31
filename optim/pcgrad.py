# Credits: https://github.com/tomtang110/multi-task_loss_optimizer/blob/master/pcgrad_fn.py

import torch
import random
import copy
import numpy as np

EPS = 1e-5  # # 防止除零的小常数

# PCGrad需要独立计算每个任务的梯度（Algorithm 1步骤1）
def get_gradient(model, loss):
    model.zero_grad()   # 清空历史梯度

    loss.backward(retain_graph=True)    # 计算当前loss的梯度并保留计算图，确保多次反向传播时计算图不被销毁


# 将处理后的梯度重新赋给模型参数
def set_gradient(grads, optimizer, shapes):
    for group in optimizer.param_groups:
        length = 0
        for i, p in enumerate(group['params']):
            # if p.grad is None: continue
            i_size = np.prod(shapes[i])     # 参数展平后的长度
            get_grad = grads[length:length + i_size]        # 提取对应梯度段
            length += i_size
            p.grad = get_grad.view(shapes[i])   # 恢复原始形状

# PCGrad 冲突梯度处理
def pcgrad_fn(model, losses, optimizer, mode='mean'):
    # === 1. 初始化存储结构 ===
    grad_list = []  # 存储各任务梯度（展平向量）
    shapes = []     # 存储各参数的原始形状
    shares = []     # 共享参数掩码（所有任务梯度均非零的位置）

    # === 2. 计算各任务独立梯度 ===
    for i, loss in enumerate(losses):
        get_gradient(model, loss)   # 计算当前任务梯度
        grads = []
        for p in model.parameters():
            if i == 0:      # 首次迭代记录参数形状
                shapes.append(p.shape)
            # 处理None梯度（论文未显式提及，工程实现需要）
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))

        new_grad = torch.cat(grads, dim=0)  # 展平拼接
        grad_list.append(new_grad)

        # 构建共享参数掩码（论文Sec 3.3的shared parameters）
        if shares == []:    # 初始化
            shares = (new_grad != 0)
        else:       # 逐位与操作
            shares &= (new_grad != 0)
    #clear memory
    # === 3. 释放计算图内存 ===
    loss_all = 0
    for los in losses:
        loss_all += los
    loss_all.backward() # 释放计算图内存

    # === 4. PCGrad核心：梯度投影 ===
    grad_list2 = copy.deepcopy(grad_list)   # 梯度副本
    for g_i in grad_list:   # 遍历每个任务梯度
        random.shuffle(grad_list2)  # 随机排列（Algorithm 1步骤3），消除任务顺序带来的偏差
        for g_j in grad_list2:      # 遍历其他任务
            g_i_g_j = torch.dot(g_i, g_j)   # 计算点积

            # 冲突检测（论文Eq.2条件），点积小于0表示梯度方向冲突
            if g_i_g_j < 0:
                # 投影操作（论文Eq.3）
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2 + EPS)

    # === 5. 梯度聚合 ===
    grads = torch.cat(grad_list, dim=0)
    grads = grads.view(len(losses), -1)
    if mode == 'mean':  # 共享参数取平均（论文Sec 3.3）
        grads_share = grads * shares.float()    # 掩码应用

        grads_share = grads_share.mean(dim=0)
        grads_no_share = grads * (1 - shares.float())
        grads_no_share = grads_no_share.sum(dim=0)

        grads = grads_share + grads_no_share
    else:   # 默认求和（原始PCGrad）
        grads = grads.sum(dim=0)

    # === 6. 更新模型梯度 ===
    set_gradient(grads, optimizer, shapes)













