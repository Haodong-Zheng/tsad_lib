import numpy as np
import torch
from tqdm import tqdm
from configs import Config
from models.dpfan import DPFAN
from utils.data_loader import load_data
from utils.metrics import calculate_metrics
from utils._tsad_adjustment import point_adjustment
from models.anomalytransformer import AnomalyTransformer
from models.couta import COUTA
from models.dsvdd import DeepSVDD
from models.modrentcn import ModernTCN
from models.ncad import NCAD
from models.fits import FITS
from models.dif import DIF
from models.itransformer import iTransformer
from models.tranad import TranAD
from models.timesnet import TimesNet
from utils.filter import filter
from utils.reco_con_dis_0802 import reco

def main(config):
    # 加载数据
    _, _, test_loader = load_data(config)

    if Config.dataset == "SDC":
        discrete_idx, continuous_idx = reco()
    elif Config.dataset == "SMAP":
        discrete_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                    , 21, 22, 23,24]      # 离散特征索引
        continuous_idx = [0]  # 连续特征索引
    elif Config.dataset == "SMD":
        discrete_idx, continuous_idx = reco()
    elif Config.dataset == "PSM":
        discrete_idx = [21]      # 离散特征索引
        continuous_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24]  # 连续特征索引
    
    # 初始化模型
    if Config.model == "DPFAN":
        model = DPFAN(
            discrete_idx = discrete_idx,
            continuous_idx = continuous_idx,
            seq_len = Config.seq_len,
            pred_len = Config.pred_len,
            enc_in = len(discrete_idx)+len(continuous_idx),  # 模型原始输入维度（特征数）
            d_model=Config.d_model,
            d_ff=Config.d_ff,
            top_k=Config.top_k,
            num_kernels=Config.num_kernels, 
            e_layers=Config.e_layers, 
            dropout=Config.dropout
        ).to(config.device)
    elif Config.model == "TimesNet":
        model = TimesNet(
            d_model=config.d_model,
            e_layers=config.e_layers,
            seq_len = config.seq_len,
            pred_len = config.pred_len,
            d_ff = config.d_ff,
            num_kernels = config.num_kernels,
            enc_in= len(discrete_idx)+len(continuous_idx),
            top_k=config.top_k,
            dropout=config.dropout
        ).to(config.device)
    elif Config.model == "AnomalyTransformer":
        model = AnomalyTransformer(
            seq_len=Config.seq_len, lr=Config.lr, epochs=Config.epochs, batch_size=Config.batch_size, epoch_steps=10, k=3,
            d_model=Config.d_model, n_heads=4, e_layers=2, n_features=len(discrete_idx)+len(continuous_idx)
        ).to(config.device)
    elif Config.model == "COUTA":
        model = COUTA(
            n_features=len(discrete_idx)+len(continuous_idx), seq_len=Config.seq_len,
                    hidden_dims=32, kernel_size=2, dropout=0.2, bias=True
        ).to(config.device)
    elif Config.model == "DIF":
        model = DIF(
            n_features=len(discrete_idx)+len(continuous_idx), n_hidden=Config.d_model, seq_len=Config.seq_len, n_output=len(discrete_idx)+len(continuous_idx), bias=False
        ).to(config.device)
    elif Config.model == "DeepSVDD":
        model = DeepSVDD(
            n_features=len(discrete_idx)+len(continuous_idx), seq_len=Config.seq_len, rep_dim=64, 
                 hidden_dims='512', n_heads=1, d_model=Config.d_model, 
                 dropout=Config.dropout
        ).to(config.device)
    elif Config.model == "ModernTCN":
        model = ModernTCN(
            seq_len=Config.seq_len, n_features=len(discrete_idx)+len(continuous_idx), patch_size=12, patch_stride=6, stem_ratio=2, downsample_ratio=6, ffn_ratio=1,
                 num_blocks=[1], large_size=[31], small_size=[3], dims=[6], dw_dims=[6],
                 small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.0,
                 use_multi_scale=False, revin=True, affine=True, subtract_last=False, individual=False
        ).to(config.device)
    elif Config.model == "NCAD":
        model = NCAD(
            n_features=len(discrete_idx)+len(continuous_idx), seq_len=Config.seq_len, 
                 rep_dim=128, hidden_dims='32,32,32,32', 
                 kernel_size=5, dropout=Config.dropout, activation='ReLU'
        ).to(config.device)
    elif Config.model == "FITS":
        model = FITS(
            orig_seq_len=Config.seq_len,DSR=4,individual=False,
            enc_in=len(discrete_idx)+len(continuous_idx),cut_freq=15
        ).to(config.device)
    elif Config.model == "iTransformer":
        model = iTransformer(
            enc_in=len(discrete_idx)+len(continuous_idx), seq_len=Config.seq_len, d_model=Config.d_model, n_heads=3, e_layers=Config.e_layers, d_ff=128,
                 dropout=Config.dropout
        ).to(config.device)
    elif Config.model == "TranAD":
        model = TranAD(
            feats=len(discrete_idx)+len(continuous_idx)
        ).to(config.device)
    
    # 加载训练好的模型
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    
    # 收集结果
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="测试中")
        for inputs, labels in test_bar:
            inputs = inputs.to(config.device)
            
            # 前向传播
            recon = model(inputs)
            
            # 计算每个时间点的MSE (batch, window, features)
            mse_per_timestep = torch.mean((recon - inputs) ** 2, dim=-1)  # [B, T]
            
            # 只取每个窗口的第一个时间点的MSE作为异常分数
            first_timestep_scores = mse_per_timestep[:, 0]  # [B]
            
            # 收集当前batch的结果
            all_scores.extend(first_timestep_scores.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 转换结果
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = calculate_metrics(
        all_labels,
        all_scores
    )
    
    adj_scores = point_adjustment(all_labels, all_scores)
    adj_metrics = calculate_metrics(all_labels, adj_scores)
    
    # 打印结果
    # print("\n测试结果:")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1 Score: {metrics['f1']:.4f}")
    # print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    print("\n测试结果:")
    print(f"Precision: {adj_metrics['precision']:.4f}")
    print(f"Recall: {adj_metrics['recall']:.4f}")
    print(f"F1 Score: {adj_metrics['f1']:.4f}")
    print(f"AUC-ROC: {adj_metrics['auc_roc']:.4f}")
    
    # 保存结果
    with open(f'{Config.dataset}_results.txt', 'w') as f:
        # f.write("测试结果:\n")
        # f.write(f"Precision: {metrics['precision']:.4f}\n")
        # f.write(f"Recall: {metrics['recall']:.4f}\n")
        # f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        # f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n\n")
        
        f.write("测试结果:\n")
        f.write(f"Precision: {adj_metrics['precision']:.4f}\n")
        f.write(f"Recall: {adj_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {adj_metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {adj_metrics['auc_roc']:.4f}\n")
    
    print(f"结果已保存到 {Config.dataset}_results.txt")

if __name__ == "__main__":
    main(Config)
