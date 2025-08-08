# unirhogad_project/pretrain.py

import torch
import os
import dgl
import yaml
import argparse
from argparse import Namespace
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

    parser = argparse.ArgumentParser("GraphMAE_PAA Pre-training")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file for the experiment.")
    parser.add_argument('--device', type=str, default=None, help="Override the device setting in the config file.")
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

def run_pretraining():
    """
    执行预训练流程。
    """

    cmd_args = get_pretrain_args()
    with open(cmd_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = Namespace(**config_dict)
    if cmd_args.device is not None:
        args.device = cmd_args.device

    try:
        # 对所有可能使用科学记数法的浮点数参数进行转换
        args.lr = float(args.lr)
        args.pretrain_lr = float(args.pretrain_lr)
        args.l2 = float(args.l2)
    except (ValueError, TypeError) as e:
        print(f"Error: Failed to convert one or more config parameters to their expected numeric types.")
        print(f"Please check your YAML file for correct formatting. Original error: {e}")
        return # 转换失败则直接退出
        
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
        batch_size=args.pretrain_batch_size, 
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
    print(f"Building GraphMAE_PAA with {args.pretrain_encoder_type.upper()} encoder and {args.pretrain_decoder_type.upper()} decoder...")
    model = GraphMAE_PAA(
        in_dim=dataset.in_dim,
        hid_dim=args.pretrain_hid_dim,
        encoder_num_layer=args.pretrain_encoder_num_layer,
        decoder_num_layer=args.pretrain_decoder_num_layer,
        encoder_type=args.pretrain_encoder_type,
        decoder_type=args.pretrain_decoder_type,
        mask_ratio=args.mask_ratio,
        replace_ratio=args.replace_ratio,
        drop_edge_rate=args.drop_edge_rate,
        loss_fn=args.loss_fn,
        alpha_l=args.alpha_l,
        anomaly_generator=anomaly_generator,
        w_contrastive=args.w_contrastive,
        w_recon=args.w_recon,
    ).to(device)

    # 3. 设置优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    print(f"Pre-training for {args.pretrain_epochs} epochs on device '{device}'...")
    
    # 4. 训练循环
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        epoch_loss = []
        
        # 使用tqdm显示进度
        for batched_graph in tqdm(pretrain_loader, desc=f"Epoch {epoch}/{args.pretrain_epochs}", leave=False):
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

    print("\n--- Starting Linear Evaluation of Pre-trained Embeddings ---")
    
    # 5. 获取预训练嵌入
    model.eval()
    with torch.no_grad():
        # 对于单图，只有一个图；对于多图，我们需要批处理所有图
        if dataset.is_single_graph:
            g = dataset.graph_list[0].to(device)
            embeds = model.embed(g, g.ndata['feature']).cpu()
            labels = g.ndata['node_label'].cpu()
        else: # Multi-graph
            # 创建一个临时的 DataLoader 来批处理所有图
            full_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pretrain
            )
            embeds_list = []
            labels_list = []
            print("  - Extracting embeddings for all graphs in the dataset...")
            for batched_graph in tqdm(full_loader):
                batched_graph = batched_graph.to(device)
                # 确保图中有节点标签
                if 'node_label' in batched_graph.ndata:
                    embeds_list.append(model.embed(batched_graph, batched_graph.ndata['feature']).cpu())
                    labels_list.append(batched_graph.ndata['node_label'].cpu())
            embeds = torch.cat(embeds_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

    # 2. 获取正确的训练/验证/测试掩码
    print("  - Preparing data splits for evaluation...")
    dataset.prepare_split(trial_id=0, seed=args.seed) # 确保划分已生成
    
    if dataset.is_single_graph:
        train_mask = dataset.split_masks[0]['train']['n']
        val_mask = dataset.split_masks[0]['val']['n']
        test_mask = dataset.split_masks[0]['test']['n']
    else: # Multi-graph
        # 将图级别的划分掩码，广播到节点级别
        g_batched = dgl.batch(dataset.graph_list)
        
        train_g_ids = dataset.split_masks[0]['train']['g']
        val_g_ids = dataset.split_masks[0]['val']['g']
        test_g_ids = dataset.split_masks[0]['test']['g']
        
        train_mask = torch.zeros(g_batched.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(g_batched.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(g_batched.num_nodes(), dtype=torch.bool)
        
        # g.batch_num_nodes() 返回一个列表，其中每个元素是对应图的节点数
        nodes_per_graph = g_batched.batch_num_nodes().tolist()
        node_offsets = [0] + np.cumsum(nodes_per_graph).tolist()
        
        for g_id in train_g_ids:
            start, end = node_offsets[g_id], node_offsets[g_id+1]
            train_mask[start:end] = True
        for g_id in val_g_ids:
            start, end = node_offsets[g_id], node_offsets[g_id+1]
            val_mask[start:end] = True
        for g_id in test_g_ids:
            start, end = node_offsets[g_id], node_offsets[g_id+1]
            test_mask[start:end] = True

    # 3. 训练和评估线性分类器
    X_train, y_train = embeds[train_mask], labels[train_mask]
    X_val, y_val = embeds[val_mask], labels[val_mask]
    X_test, y_test = embeds[test_mask], labels[test_mask]

    print(f"  - Training logistic regression on {X_train.shape[0]} samples...")
    classifier = LogisticRegression(solver='liblinear', random_state=args.seed)
    classifier.fit(X_train.numpy(), y_train.numpy())

    print("--- Linear Evaluation Results ---")
    # 在测试集上评估
    test_probs = classifier.predict_proba(X_test.numpy())[:, 1]
    auroc = roc_auc_score(y_test.numpy(), test_probs)
    auprc = average_precision_score(y_test.numpy(), test_probs)
    
    print(f"  - Node AUROC: {auroc:.4f}")
    print(f"  - Node AUPRC: {auprc:.4f}")

    # =================================================================
    #   核心修复点：重构图任务线性评估的池化逻辑
    # =================================================================
    if not dataset.is_single_graph and 'g' in dataset.labels_have:
        print("  - Evaluating on Graph-level task...")
        
        full_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pretrain
        )
        graph_embeds_list = []
        print("    - Generating graph-level embeddings...")
        with torch.no_grad():
            for batched_graph in tqdm(full_loader):
                batched_graph = batched_graph.to(device)
                
                # 1. 获取节点嵌入
                node_embeds = model.embed(batched_graph, batched_graph.ndata['feature'])
                
                # 2. (修复) 先将节点嵌入用一个临时键名存入图中
                temp_feat_name = "_temp_embeds_for_pooling"
                batched_graph.ndata[temp_feat_name] = node_embeds
                
                # 3. (修复) 使用字符串键来调用池化函数
                graph_embeds = dgl.mean_nodes(batched_graph, feat=temp_feat_name)
                
                # 4. (修复) 操作完成后，清理临时特征
                del batched_graph.ndata[temp_feat_name]
                
                graph_embeds_list.append(graph_embeds.cpu())
        
        graph_embeds = torch.cat(graph_embeds_list, dim=0)
        graph_labels = dataset.graph_labels
        
        # 2. 获取图级别的划分
        train_g_ids = dataset.split_masks[0]['train']['g']
        test_g_ids = dataset.split_masks[0]['test']['g']
        
        X_train_g, y_train_g = graph_embeds[train_g_ids], graph_labels[train_g_ids]
        X_test_g, y_test_g = graph_embeds[test_g_ids], graph_labels[test_g_ids]

        # 3. 训练和评估图分类器
        print(f"    - Training logistic regression on {X_train_g.shape[0]} graph samples...")
        graph_classifier = LogisticRegression(solver='liblinear', random_state=args.seed)
        graph_classifier.fit(X_train_g.numpy(), y_train_g.numpy())
        
        test_probs_g = graph_classifier.predict_proba(X_test_g.numpy())[:, 1]
        graph_auroc = roc_auc_score(y_test_g.numpy(), test_probs_g)
        graph_auprc = average_precision_score(y_test_g.numpy(), test_probs_g)
        
        print(f"  - Graph AUROC: {graph_auroc:.4f}")
        print(f"  - Graph AUPRC: {graph_auprc:.4f}")

    # 4. 保存模型
    # 构建一个有意义的文件名
    # 清理数据集名称中的斜杠，替换为下划线
    dataset_name_clean = args.dataset.replace('/', '_')
    
    # 构建一个包含所有关键信息的、有意义的文件名
    model_filename = (
        f"{dataset_name_clean}_"
        f"enc_{args.pretrain_encoder_type}{args.pretrain_encoder_num_layer}_"
        f"dec_{args.pretrain_decoder_type}{args.pretrain_decoder_num_layer}_"
        f"e{args.pretrain_epochs}_"
        f"h{args.pretrain_hid_dim}_"
        f"loss_{args.loss_fn}.pt"
    )
    
    save_path = os.path.join(args.save_dir, model_filename)
    
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--- Pre-training finished. Model saved to {save_path} ---")
        
    return model, save_path

if __name__ == '__main__':
    run_pretraining()