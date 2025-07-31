# unirhogad_project/pretrain.py

import torch
import os
import dgl
import argparse
from tqdm import tqdm
import numpy as np
# 导入我们项目中的模块
from data.dataset_loader import UniGADDataset
from utils.misc import set_seed, get_current_lr

from data.anomaly_generator import AnomalyGenerator
from models.pretrain_model import GraphMAE_PAA 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def get_pretrain_args():
    """
    为预训练脚本解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="GraphMAE_PAA Pre-training for Uni-RHO-GAD")
    
    # --- 路径与设备 ---
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'reddit', 'mutag/dgl/mutag0')")
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory where data is stored')
    parser.add_argument('--save_dir', type=str, default='./pretrained', help='Directory to save the pretrained models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 训练控制 ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--epochs', type=int, default=100, help="Number of pre-training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for pre-training")
    
    # --- 模型与优化器 ---
    parser.add_argument('--w_contrastive', type=float, default=0.5, help="Weight for the contrastive loss in PAA pre-training.")
    parser.add_argument('--encoder_type', type=str, default='gcn', help="GNN type for the encoder (e.g., 'gcn', 'gin', 'bwgnn')")
    parser.add_argument('--decoder_type', type=str, default='gcn', help="Type for the decoder (e.g., 'gcn', 'mlp')")
    parser.add_argument('--hid_dim', type=int, default=64, help="Hidden dimension of the GNN")
    parser.add_argument('--encoder_num_layer', type=int, default=2, help="Number of layers in the GNN encoder")
    parser.add_argument('--decoder_num_layer', type=int, default=1, help="Number of layers in the GNN decoder")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for pre-training")
    parser.add_argument('--l2', type=float, default=0.0, help="L2 weight decay")
    parser.add_argument('--scheduler_step', type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help="Gamma for learning rate scheduler")
    
    # --- GraphMAE_PAA 特定参数 ---
    parser.add_argument('--mask_ratio', type=float, default=0.5, help="Ratio of nodes to mask")
    parser.add_argument('--replace_ratio', type=float, default=0.1, help="Ratio of masked nodes to be replaced by random features")
    parser.add_argument('--drop_edge_rate', type=float, default=0.2, help="Ratio of edges to drop")
    parser.add_argument('--loss_fn', type=str, default='sce', help="Loss function ('mse' or 'sce')")
    parser.add_argument('--alpha_l', type=float, default=2.0, help="Alpha parameter for SCE loss")
    
    return parser.parse_args()

def evaluate_embeddings_linearly(embeddings, labels, train_mask, test_mask):
    """使用逻辑回归进行线性评估"""
    X_train = embeddings[train_mask].cpu().numpy()
    y_train = labels[train_mask].cpu().numpy()
    X_test = embeddings[test_mask].cpu().numpy()
    y_test = labels[test_mask].cpu().numpy()

    lr = LogisticRegression(solver='liblinear', class_weight='balanced')
    lr.fit(X_train, y_train)
    
    probs = lr.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    
    return auroc, auprc

def collate_pretrain(samples):
    """一个简化的collate函数，只处理图，因为预训练不需要标签。"""
    graphs, _ = map(list, zip(*samples))
    return dgl.batch(graphs)

def run_pretraining(args):
    """
    执行预训练流程。
    """
    print("--- Starting GraphMAE_PAA Pre-training ---")
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. 加载数据集
    print(f"Loading dataset: {args.dataset} from {args.data_dir}")
    try:
        dataset = UniGADDataset(name=args.dataset, data_dir=args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset exists and the paths are correct.")
        return
    
    # 预训练使用整个数据集
    # 注意：对于单图，Dataset会返回原始大图，DataLoader只会产生一个批次
    pretrain_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_pretrain,
        num_workers=4 if torch.cuda.is_available() else 0 # 加速数据加载
    )

    # --- 新增：为预训练初始化 AnomalyGenerator ---
    anomaly_generator = None
    if dataset.is_single_graph:
        # 预训练时，我们可能在多个小图上进行，也可能在一个大图上
        # 假设我们只为单图场景启用此功能
        anomaly_generator = AnomalyGenerator(dataset.graph_list[0])

    # 2. 初始化模型 (使用我们完善后的GraphMAE_PAA)
    print("Building GraphMAE_PAA with Pre-training Adversarial Augmentation (PAA)...")
    print(f"Building GraphMAE_PAA with {args.encoder_type.upper()} encoder and {args.decoder_type.upper()} decoder...")
    model = GraphMAE_PAA(
        in_dim=dataset.in_dim,
        hid_dim=args.hid_dim,
        encoder_num_layer=args.encoder_num_layer,
        decoder_num_layer=args.decoder_num_layer,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        mask_ratio=args.mask_ratio,
        replace_ratio=args.replace_ratio,
        drop_edge_rate=args.drop_edge_rate,
        loss_fn=args.loss_fn,
        alpha_l=args.alpha_l,
        anomaly_generator=anomaly_generator,
        w_contrastive=args.w_contrastive
    ).to(device)

    # 3. 设置优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    print(f"Pre-training for {args.epochs} epochs on device '{device}'...")
    
    # 4. 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []
        
        # 使用tqdm显示进度
        for batched_graph in tqdm(pretrain_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            batched_graph = batched_graph.to(device)
            # 确保图中有特征
            if 'feature' not in batched_graph.ndata:
                print("Warning: 'feature' not found in batched_graph.ndata. Skipping batch.")
                continue
            features = batched_graph.ndata['feature']
            
            loss = model(batched_graph, features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch:03d} | Avg Pre-train Loss: {avg_loss:.4f} | LR: {get_current_lr(optimizer):.6f}")

    # --- 新增：评估阶段 ---
    print("\n--- Starting Linear Evaluation of Pre-trained Embeddings ---")
    model.eval()
    with torch.no_grad():
        # 假设 dataset.graph_list[0] 是我们的图
        g = dataset.graph_list[0].to(device)
        embeddings = model.embed(g, g.ndata['feature'])
    
    # 获取标签和划分
    labels = g.ndata['node_label']
    # 需要一个方法来获取划分，这里我们简化一下
    # 假设 dataset 已经有了划分信息
    dataset.prepare_split(trial_id=0, seed=args.seed) # 使用固定的划分
    train_mask = dataset.split_masks[0]['train']['n']
    test_mask = dataset.split_masks[0]['test']['n']

    auroc, auprc = evaluate_embeddings_linearly(embeddings, labels, train_mask, test_mask)
    
    print(f"--- Linear Evaluation Results ---")
    print(f"  - Node AUROC: {auroc:.4f}")
    print(f"  - Node AUPRC: {auprc:.4f}")

    # 5. 保存模型
    # 构建一个有意义的文件名
    # 清理数据集名称中的斜杠，替换为下划线
    dataset_name_clean = args.dataset.replace('/', '_')
    
    # 构建一个包含所有关键信息的、有意义的文件名
    model_filename = (
        f"{dataset_name_clean}_"
        f"enc_{args.encoder_type}{args.encoder_num_layer}_"
        f"dec_{args.decoder_type}{args.decoder_num_layer}_"
        f"e{args.epochs}_"
        f"h{args.hid_dim}_"
        f"loss_{args.loss_fn}.pt"
    )
    
    save_path = os.path.join(args.save_dir, model_filename)
    
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--- Pre-training finished. Model saved to {save_path} ---")
        
    return model, save_path

if __name__ == '__main__':
    args = get_pretrain_args()
    run_pretraining(args)