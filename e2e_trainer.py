# unirhogad_project/e2e_trainer.py

import torch
import torch.nn.functional as F
import dgl.function as fn
from tqdm import tqdm
import numpy as np
import dgl
import pprint
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils.misc import FocalLoss
from predictors.unirhogad_predictor import Uni_RHO_GAD_Predictor
from data.anomaly_generator import AnomalyGenerator
from utils.misc import get_current_lr

class Trainer:
    def __init__(self,  model, train_loader, val_loader, test_loader, args):
        """
        初始化训练器。

        Args:
            model: 主模型 Uni_RHO_GAD_Predictor。
            train_loader, val_loader, test_loader: 数据加载器。
            args: 命令行参数。
        """
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = args.device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        # 初始化时不设置alpha，因为我们将在forward时动态传入
        self.classification_loss_fn = FocalLoss(gamma=2, alpha=None, reduction="mean")


    def _get_best_f1(self, labels, probs):
        """通过搜索阈值找到最佳的宏F1分数"""
        best_f1 = 0
        # 确保标签和概率是numpy数组
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
        
        for thres in np.linspace(0.05, 0.95, 19):
            preds = (probs > thres).astype(int)
            best_f1 = max(best_f1, f1_score(labels, preds, average='macro', zero_division=0))
        return best_f1

    def _compute_metrics(self, labels, probs):
        """计算所有评估指标"""
        # 确保标签和概率是numpy数组
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
        
        # 检查标签是否只有一个类别，这会导致AUROC计算错误
        if len(np.unique(labels)) < 2:
            print("Warning: Only one class present in labels. Metrics will be trivial.")
            return {'AUROC': 0.5, 'AUPRC': np.mean(labels), 'MacroF1': f1_score(labels, probs > 0.5, average='macro', zero_division=0)} # 返回一个无意义但安全的默认值
            
        return {
            'AUROC': roc_auc_score(labels, probs),
            'AUPRC': average_precision_score(labels, probs),
            'MacroF1': self._get_best_f1(labels, probs)
        }

    def _calculate_loss_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        根据一个批次的标签动态计算类别权重。
        权重 = 1 / (类别频率)
        """
        if labels.numel() == 0:
            return None

        # 计算每个类别的样本数
        class_counts = torch.bincount(labels, minlength=2).float()
        
        # 如果某个类别不存在，则其权重为0，避免除零错误
        # 权重计算：总样本数 / (类别数 * 类别样本数) 是一种常见的归一化方法
        # 或者更简单的：1 / 类别频率
        total_samples = class_counts.sum()
        if total_samples == 0:
            return None
        
        # 避免除以零
        class_freq = class_counts / total_samples
        weights = 1.0 / (class_freq + 1e-6) # 加一个小的epsilon防止除零
        
        # 归一化权重，使其和为类别数 (可选，但推荐)
        weights = weights / weights.sum() * 2.0
        
        return weights.to(self.device)


    def train(self):
        """执行完整的训练、验证和早停流程"""
        best_val_metric = -1.0
        patience_counter = 0
        best_test_metrics = None
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            # --- 训练阶段 ---
            self.model.train()
            # 预训练模型始终处于评估模式
            self.model.pretrain_model.eval() 
            
            epoch_total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
            for data in pbar:
                # collate_fn 返回 (原始图/批处理图, 标签字典, 任务图字典)
                original_graph, batched_labels, task_graphs = data
                
                # 将所有数据移动到设备
                task_graphs = {k: v.to(self.device) for k, v in task_graphs.items()}
                batched_labels = {k: v.to(self.device) for k, v in batched_labels.items()}
                if original_graph:
                    original_graph = original_graph.to(self.device)
                
                # 准备 normal_masks 用于单类损失
                normal_masks = {k: (v == 0) for k, v in batched_labels.items() if v.numel() > 0}

                all_logits, shared_losses = self.model(
                    batched_inputs=task_graphs,
                    original_graph=original_graph,
                    normal_masks=normal_masks
                )

                # 计算总损失
                loss_cls = 0
                # 遍历所有 cross_modes 的输出
                for mode_name, logits_dict in all_logits.items():
                    # 遍历该模式下的每个任务输出
                    for task, logits in logits_dict.items():
                        if task in batched_labels and batched_labels[task].numel() > 0:

                            labels = batched_labels[task]
                            # 计算动态类别权重
                            class_weights = self._calculate_loss_weights(labels)
                            loss_cls += self.classification_loss_fn(logits, labels, weight=class_weights)

                            
                
                # 加权组合所有损失
                loss = (self.args.w_classification * loss_cls +
                        self.args.w_gna * shared_losses.get('gna', 0) +
                        self.args.w_one_class * shared_losses.get('one_class', 0))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            avg_train_loss = epoch_total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            print(f"Epoch {epoch:03d} | Avg Train Loss: {avg_train_loss:.4f}")

            # --- 验证阶段 ---
            val_metrics = self.evaluate('val')
            
            # --- 早停逻辑 ---
            current_val_metric = -1.0
            monitor_key_found = False

            # 定义一个首选的监控顺序
            preferred_modes = [mode.replace('2', '_to_') for mode in self.args.cross_modes.split(',')]
            preferred_tasks = ['n', 'g', 'e']

            for mode in preferred_modes:
                if mode in val_metrics and val_metrics[mode]: # 确保 mode 存在且其结果不为空
                    for task in preferred_tasks:
                        if task in val_metrics[mode] and self.args.metric in val_metrics[mode][task]:
                            current_val_metric = val_metrics[mode][task][self.args.metric]
                            print(f"Epoch {epoch:03d} | Val Metric ({self.args.metric} on {mode}/{task}): {current_val_metric:.4f}")
                            monitor_key_found = True
                            break # 找到了，跳出内层循环
                if monitor_key_found:
                    break # 找到了，跳出外层循环
            
            if not monitor_key_found:
                print("Warning: Could not get any valid validation metric for early stopping. Skipping check.")

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
                print("New best validation score! Evaluating on test set...")
                best_test_metrics = self.evaluate('test')
                # 打印当前最佳的测试结果
                for mode, metrics_dict in best_test_metrics.items():
                    print(f"  > Test Results for mode [{mode}]:")
                    for task, metrics in metrics_dict.items():
                        print(f"    - Task [{task}]: {', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        
        total_time_cost = time.time() - start_time
        print("\n--- Training Finished ---")
        return best_test_metrics, total_time_cost

    def evaluate(self, split='val'):
        self.model.eval()

        loader = self.val_loader if split == 'val' else self.test_loader

        
        # 初始化用于存储所有预测结果的结构
        all_preds = {mode: {task: [] for task in head.output_route} for mode, head in self.model.fusion_heads.items()}
        all_labels = {mode: {task: [] for task in head.output_route} for mode, head in self.model.fusion_heads.items()}


        with torch.no_grad():
            for data in tqdm(loader, desc=f"Evaluating on {split} set", leave=False):
                
                original_graph, batched_labels, batched_inputs = data
                batched_inputs = {k: v.to(self.device) for k, v in batched_inputs.items()}
                
                if original_graph:
                    original_graph = original_graph.to(self.device)

                all_logits, _ = self.model(
                    batched_inputs=batched_inputs,
                    original_graph=original_graph
                    )


                for mode_name, logits_dict in all_logits.items():
                    for task, logits in logits_dict.items():
                        if task in batched_labels and batched_labels[task].numel() > 0:
                            probs = F.softmax(logits, dim=1)[:, 1]
                            # 确保 mode_name 和 task 在 all_preds 中存在
                            if mode_name in all_preds and task in all_preds[mode_name]:
                                all_preds[mode_name][task].append(probs.cpu())
                                all_labels[mode_name][task].append(batched_labels[task].cpu())
        
        # 计算最终指标
        final_metrics = {}
        for mode_name in all_preds:
            final_metrics[mode_name] = {}
            for task in all_preds[mode_name]:
                if all_preds[mode_name][task]:
                    preds_cat = torch.cat(all_preds[mode_name][task])
                    labels_cat = torch.cat(all_labels[mode_name][task])
                    final_metrics[mode_name][task] = self._compute_metrics(labels_cat, preds_cat)

        
        return final_metrics
