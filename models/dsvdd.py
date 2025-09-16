# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math

class DeepSVDD(nn.Module):
    def __init__(self, n_features, seq_len, rep_dim, 
                 hidden_dims, n_heads, d_model, dropout):
        super(DeepSVDD, self).__init__()
        self.rep_dim = rep_dim
        self.c = None
        
        self.net = TSTransformerEncoder(
            n_features=n_features,
            n_output=rep_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_hidden=hidden_dims,
            dropout=dropout
        )

    def forward(self, x):
        """前向传播，返回特征表示"""
        # 输入形状: (batch_size, seq_len, n_features)
        return self.net(x)
    
    def init_center(self, train_loader, eps=0.1):
        """初始化超球面中心点c"""
        self.eval()
        z_ = []
        
        with torch.no_grad():
            for batch in train_loader:
                representations = self.forward(batch)
                z_.append(representations)
        
        # 计算所有样本特征的平均值作为初始中心
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        
        # 避免中心点过接近零点
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        
        self.c = c
        self.train()
        return c


class TSTransformerEncoder(nn.Module):
    def __init__(self, n_features, n_output=20, seq_len=100, d_model=64,
                 n_heads=8, n_hidden='128', dropout=0.1,
                 token_encoding='linear', pos_encoding='fixed',
                 activation='GELU', bias=False):
        super(TSTransformerEncoder, self).__init__()
        self.max_len = seq_len
        self.d_model = d_model
        
        if isinstance(n_hidden, str):
            n_hidden = [int(dim) for dim in n_hidden.split(',')]
        
        # 输入投影层
        if token_encoding == 'linear':
            self.project_inp = nn.Linear(n_features, d_model, bias=bias)
        elif token_encoding == 'convolutional':
            self.project_inp = TokenEmbedding(n_features, d_model, kernel_size=3, bias=bias)
        
        # 位置编码
        if pos_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout, max_len=seq_len)
        elif pos_encoding == "fixed":
            self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout, max_len=seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=n_hidden[0] if n_hidden else d_model,
            dropout=dropout,
            activation="gelu" if activation == 'GELU' else "relu",
            batch_first=False,
            norm_first=False
        )
        
        num_layers = len(n_hidden) if n_hidden else 1
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reconstruction_layer = nn.Linear(d_model, n_features, bias=bias)
        self.activation = nn.GELU() if activation == 'GELU' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_len, n_features = X.shape
        
        # 输入投影
        projected = self.project_inp(X)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        pos_encoded = self.pos_enc(projected)  # (batch_size, seq_len, d_model)
        
        # 合并投影和位置编码
        inp = projected + pos_encoded
        
        # 调整维度为(seq_len, batch_size, d_model)
        inp = inp.permute(1, 0, 2)
        
        # Transformer处理
        output = self.transformer_encoder(inp)  # (seq_len, batch_size, d_model)
        
        # 恢复维度为(batch_size, seq_len, d_model)
        output = output.permute(1, 0, 2)
        output = self.reconstruction_layer(output)  # (batch_size, seq_len, n_features)
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


class DSVDDLoss(nn.Module):
    """损失函数"""
    def __init__(self, center, reduction='mean'):
        super(DSVDDLoss, self).__init__()
        self.c = center
        self.reduction = reduction

    def forward(self, rep):
        """计算样本表示到中心点的平方距离"""
        loss = torch.sum((rep - self.c) ** 2, dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss

class TokenEmbedding(nn.Module):
    """卷积token嵌入"""
    def __init__(self, in_features, embed_dim, kernel_size=3, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.conv(x)
        return x.permute(0, 2, 1)  # (batch, seq_len, features)


class FixedPositionalEncoding(nn.Module):
    """固定位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        # 确保位置编码维度与输入匹配
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    