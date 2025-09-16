import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from configs import Config
from models.dpfan import DPFAN
from utils.data_loader import load_data
from utils.early_stopping import EarlyStopping
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
    seed = Config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    os.makedirs("checkpoints", exist_ok=True)

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
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.patience,
        delta=0.001,
        path=config.model_save_path
    )

    # 加载数据
    train_loader, val_loader, _ = load_data(config)
    # # 验证第一个batch
    # first_batch = next(iter(train_loader))
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # 训练进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch in train_bar:
            inputs = batch.to(config.device)
            optimizer.zero_grad()
            
            # 前向传播
            recon  = model(inputs)
            loss = criterion(recon, inputs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
            for batch in val_bar:
                inputs = batch.to(config.device)
                recon = model(inputs)
                loss = criterion(recon, inputs)
                val_loss += loss.item() * inputs.size(0)
        
        # 计算平均损失
        epoch_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        # 打印epoch信息
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {epoch_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {time.time()-start_time:.2f}s")
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"早停于第 {epoch+1} 轮")
            break
    
    print("训练完成! 最佳模型已保存至:", config.model_save_path)

if __name__ == "__main__":
    main(Config)
