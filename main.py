import torch
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
import pandas as pd
import time
from functools import partial
from data.dataset_loader import UniGADDataset, collate_fn_unify
from models.pretrain_model import GraphMAE_PAA
from predictors.unirhogad_predictor import Uni_RHO_GAD_Predictor
from e2e_trainer import Trainer
from utils.misc import set_seed
from torch.utils.data import WeightedRandomSampler

def get_args():
    """解析所有命令行参数"""
    parser = argparse.ArgumentParser("Uni-RHO-GAD End-to-End Training")
    
    # --- 数据与路径参数 ---
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'reddit', 'mutag/dgl/mutag0')")
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory where data is stored')
    parser.add_argument('--pretrain_path', type=str, required=True, help='Path to the saved pretrained model weights (.pt file)')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save the final results CSV')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # =================================================================
    #   核心修改点：添加数据增强相关的命令行参数
    # =================================================================
    parser.add_argument('--use_anomaly_generation', action='store_true', 
                        help="Enable generation of artificial anomalies for single-graph tasks.")
    parser.add_argument('--use_node_aug', action='store_true', 
                        help="Enable artificial anomaly generation for node tasks (if use_anomaly_generation is set).")
    parser.add_argument('--use_edge_aug', action='store_true', 
                        help="Enable artificial anomaly generation for edge tasks (if use_anomaly_generation is set).")
    parser.add_argument('--aug_ratio', type=float, default=0.5, help="Ratio of normal samples to be perturbed into anomalies per batch.")
    parser.add_argument('--aug_num_perturb_edges', type=int, default=5, help="Number of perturbed edges to add for each artificial anomaly.")
    parser.add_argument('--aug_feature_mix_ratio', type=float, default=0.5, help="Mixing ratio for feature perturbation (beta).")
    
    # --- 训练控制参数 ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--num_trials', type=int, default=5, help="Number of trials with different seeds for robust evaluation")
    parser.add_argument('--epochs', type=int, default=200, help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--metric', type=str, default='AUROC', choices=['AUROC', 'AUPRC', 'MacroF1'], help='Metric for early stopping')
    

    # --- 模型与优化器参数 ---
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--hid_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--pretrain_encoder_type', type=str, default='gcn', help="The GNN type of the loaded pretrained encoder.")
    parser.add_argument('--pretrain_decoder_type', type=str, default='gcn', help="The type of the loaded pretrained decoder.")
    parser.add_argument('--pretrain_hid_dim', type=int, default=64, help="The hidden dim of the loaded pretrained model.")
    parser.add_argument('--pretrain_encoder_num_layer', type=int, default=2, help="The number of layers of the loaded pretrained encoder.")
    parser.add_argument('--pretrain_decoder_num_layer', type=int, default=1, help="The number of layers of the loaded pretrained decoder.")
    
    parser.add_argument('--base_gnn_layers', type=int, default=2)
    parser.add_argument('--final_mlp_layers', type=int, default=2)
    parser.add_argument('--gna_proj_dim', type=int, default=128)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--norm', type=str, default='layernorm')

    # --- Uni-RHO-GAD 特定参数 ---
    parser.add_argument('--all_tasks', type=str, default='neg', help="All possible tasks in the dataset (e.g., 'n', 'ng', 'neg')")
    parser.add_argument('--cross_modes', type=str, default='ng2ng,n2ng,g2ng', help="Comma-separated list of cross modes to evaluate (e.g., 'ng2ng,n2ng')")
    parser.add_argument('--w_one_class', type=float, default=1.0, help="Weight for one-class loss")
    parser.add_argument('--w_gna', type=float, default=1.0, help="Weight for GNA contrastive loss")
    parser.add_argument('--w_classification', type=float, default=1.0, help="Weight for supervised classification loss")
    
    return parser.parse_args()


def main(args):
    """主执行函数"""
    set_seed(args.seed)
    
    # 1. ==================== 数据加载 ====================
    print(f"--- Loading dataset: {args.dataset} ---")
    dataset = UniGADDataset(name=args.dataset, data_dir=args.data_dir)
    
    # 为每个划分创建 DataLoader
    # 我们只在一个 trial 上运行，所以 trial_id=0
    dataset.prepare_split(trial_id=0, seed=args.seed)
    train_subset = dataset.get_subset('train', trial_id=0)
    val_subset = dataset.get_subset('val', trial_id=0)
    test_subset = dataset.get_subset('test', trial_id=0)
    
    # collate_fn = partial(collate_fn_unify, sampler=dataset.sampler)

    # =================================================================
    #   核心修改点：使用 partial 绑定所有需要的参数
    # =================================================================
    anomaly_generator_instance = None
    if args.use_anomaly_generation and dataset.is_single_graph:
        # 从 dataset 中获取已经初始化好的 generator
        anomaly_generator_instance = dataset.anomaly_generator

    collate_with_aug = partial(
        collate_fn_unify, 
        sampler=dataset.sampler, 
        anomaly_generator=anomaly_generator_instance,
        aug_ratio=args.aug_ratio,
        num_perturb_edges=args.aug_num_perturb_edges,
        feature_mix_ratio=args.aug_feature_mix_ratio,
        use_node_aug=args.use_node_aug,
        use_edge_aug=args.use_edge_aug
    )

    # train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ================================================================
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_aug)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_aug)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_aug)

    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}")

    # 2. ==================== 模型构建 ====================
    print("--- Building models ---")
    # a. 加载预训练的GraphMAE_PAA模型 (仅用于特征提取)
    # 注意：这里的参数需要与你预训练时使用的参数大致匹配
    print(f"Instantiating pretrained model architecture: enc={args.pretrain_encoder_type}, dec={args.pretrain_decoder_type}, hid={args.pretrain_hid_dim}")
    pretrain_model = GraphMAE_PAA(
        in_dim=dataset.in_dim,
        hid_dim=args.pretrain_hid_dim,
        encoder_num_layer=args.pretrain_encoder_num_layer,
        decoder_num_layer=args.pretrain_decoder_num_layer,
        encoder_type=args.pretrain_encoder_type,
        decoder_type=args.pretrain_decoder_type
    )
    print(f"Loading pretrained weights from {args.pretrain_path}")
    pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location=args.device, weights_only=True))
    pretrain_model.eval()  # 切换到评估模式

    # 创建特征适配器，以处理预训练输出维度和主模型输入维度不匹配的情况
    pretrain_output_dim = pretrain_model.embed_dim
    if pretrain_output_dim != args.hid_dim:
        print(f"Dimension mismatch: Adapting pretrained output from {pretrain_output_dim} to main model dim {args.hid_dim}.")
        feature_adapter = nn.Linear(pretrain_output_dim, args.hid_dim)
    else:
        feature_adapter = nn.Identity()

    # b. 构建我们的主模型 Uni_RHO_GAD_Predictor
    # 传入 cross_modes 列表
    cross_modes_list = args.cross_modes.split(',')
    all_tasks_list = list(args.all_tasks)
    model = Uni_RHO_GAD_Predictor(
        pretrain_model=pretrain_model,
        feature_adapter=feature_adapter,
        is_single_graph=dataset.is_single_graph,
        embed_dims=args.hid_dim,
        num_classes=2, # 异常/正常
        all_tasks=all_tasks_list,
        cross_modes=cross_modes_list,
        base_gnn_layers=args.base_gnn_layers,
        final_mlp_layers=args.final_mlp_layers,
        gna_projection_dim=args.gna_proj_dim,
        dropout_rate=args.dropout,
        activation=args.activation,
        residual=args.residual,
        norm=args.norm
    )

    # 3. ==================== 训练启动 ====================
    print("--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args
    )
    
    print("--- Starting end-to-end training ---")
    final_test_metrics, total_time_cost = trainer.train()
    print("--- Training finished ---")

    # 4. ==================== 结果保存 ====================
    if final_test_metrics:
        all_results = []
        for mode, metrics_dict in final_test_metrics.items():
            result_row = {
                'dataset': args.dataset,
                'pretrain_model': os.path.basename(args.pretrain_path),
                'cross_mode': mode,
                'hid_dim': args.hid_dim,
                'lr': args.lr,
                'time_cost': total_time_cost / len(final_test_metrics)
            }
            for task, metrics in metrics_dict.items():
                for metric_name, value in metrics.items():
                    result_row[f'{task}_{metric_name}'] = value
            all_results.append(result_row)
        
        results_df = pd.DataFrame(all_results)
        
        dataset_name_clean = args.dataset.replace('/', '_')
        timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
        filename = f"results_{dataset_name_clean}_{timestamp}.csv"
        save_path = os.path.join(args.results_dir, filename)
        os.makedirs(args.results_dir, exist_ok=True)
        
        results_df.to_csv(save_path, index=False)
        print("\n--- Final Test Results ---")
        print(results_df)
        print(f"\nResults saved to: {save_path}")
    else:
        print("Training finished, but no valid test metrics were generated.")

if __name__ == '__main__':
    args = get_args()
    main(args)