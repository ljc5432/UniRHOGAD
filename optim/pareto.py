# Credits: https://github.com/tomtang110/multi-task_loss_optimizer/blob/master/Pareto_fn.py

from scipy.optimize import minimize
from scipy.optimize import nnls
import numpy as np
import torch

# 解决带约束的最小二乘问题（Active Set Method）
def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    # 创建单位矩阵 A (K×K)，用于最小二乘问题
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    # 用非负最小二乘（NNLS）求解初始点：min ||Ax - b||² s.t. x ≥ 0
    x0, _ = nnls(A, b)

    # 定义目标函数：||Ax - b||₂
    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    # 添加等式约束：Σx + Σc = 1（确保权重和为1）
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    # 定义边界条件：x ≥ 0
    bounds = [[0., None] for _ in range(len(hat_w))]
    # 用SLSQP算法求解带约束优化问题
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    # 计算最终权重：new_w = x* + c
    new_w = min_out.x + c
    return new_w

# 核心Pareto优化步骤（求解KKT条件）
def pareto_step(w, c, G):
    """
    ref:http://ofey.me/papers/Pareto.pdf
    K : the number of task
    M : the dim of NN's params
    :param W: # (K,1)
    :param C: # (K,1)
    :param G: # (K,M)
    :return:
    """
    # 计算Gram矩阵 G·Gᵀ (K×K)
    GGT = np.matmul(G, np.transpose(G))  # (K, K)
    e = np.mat(np.ones(np.shape(w)))  # (K, 1)
    # 构造增广矩阵 M = [G·Gᵀ, e; eᵀ, 0] ((K+1)×(K+1))
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((np.transpose(e), np.mat(np.zeros((1, 1)))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    # 构造向量 z = [-G·Gᵀ·c; 1-Σc] (K+1,1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    # 求解最小二乘问题：(MᵀM)⁻¹Mᵀz（Moore-Penrose伪逆）
    hat_w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(M), M)), M), z)  # (K+1, 1)
    # 去掉最后一个元素（拉格朗日乘子）
    hat_w = hat_w[:-1]  # (K, 1)
    # 将矩阵转换为向量
    hat_w = np.reshape(np.array(hat_w), (hat_w.shape[0],))  # (K,)
    c = np.reshape(np.array(c), (c.shape[0],))  # (K,)
    # 调用ASM获得最终权重
    new_w = ASM(hat_w, c)
    return new_w

# 计算指定损失的梯度（保留计算图）
def apply_gradient(model,loss):
    model.zero_grad()
    loss.backward(retain_graph=True)

# Pareto 梯度优化
def pareto_fn(w_list, c_list, model, num_tasks, loss_list):
    grads = [[] for i in range(len(loss_list))]

    '''
        ​​梯度收集​​：
            遍历每个任务的损失
            收集模型所有参数的梯度（展平为向量）
            拼接成单个梯度向量 (M,)
            转换为NumPy数组
    '''

    for i,loss in enumerate(loss_list):
        for p in model.parameters():
            if p.grad is not None:
                grads[i].append(p.grad.view(-1))
            else:
                grads[i].append(torch.zeros_like(p).cuda(non_blocking=True).view(-1))

        grads[i] = torch.cat(grads[i],dim=-1).cpu().numpy()


    # 堆叠成梯度矩阵 G (K×M)
    grads = np.concatenate(grads,axis=0).reshape(num_tasks,-1)
    # 将权重和偏移转换为列向量
    weights = np.mat([[w] for w in w_list])
    c_mat = np.mat([[c] for c in c_list])
    # 调用pareto_step计算新权重
    new_w_list = pareto_step(weights, c_mat, grads)

    return new_w_list