import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAwareAttentionFusion(nn.Module):
    """特征感知注意力融合模块 (FAAF)"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 交叉注意力机制
        self.discrete_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout)
        self.continuous_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )


    def forward(self, discrete_feat, continuous_feat):
        """
            输入：discrete_feat (离散特征), continuous_feat (连续特征)
            步骤：
            1. 转置维度：(B, T, C) -> (T, B, C) 以满足MultiheadAttention的输入要求（序列长度在第一位）
            2. 交叉注意力：
                离散特征作为query，连续特征作为key和value -> 得到离散特征的注意力输出
                连续特征作为query，离散特征作为key和value -> 得到连续特征的注意力输出
            3. 转置回 (B, T, C)
            4. 残差连接和层归一化：将原始特征（转置后）与注意力输出相加，然后归一化
            5. 门控融合：将两个特征拼接，通过门控层生成权重，然后加权融合两个特征
            6. 前馈网络：融合后的特征经过FFN，再经过残差连接和层归一化
        """
        # 转置维度 (B, T, C) -> (T, B, C)
        discrete_feat = discrete_feat.transpose(0, 1)
        continuous_feat = continuous_feat.transpose(0, 1)
        
        # 交叉注意力
        discrete_out, _ = self.discrete_attn(
            discrete_feat, continuous_feat, continuous_feat
        )
        continuous_out, _ = self.continuous_attn(
            continuous_feat, discrete_feat, discrete_feat
        )
        
        # 转置回原始维度
        discrete_out = discrete_out.transpose(0, 1)
        continuous_out = continuous_out.transpose(0, 1)
        
        # 残差连接和层归一化
        discrete_feat = self.norm1(discrete_feat.transpose(0, 1) + discrete_out)
        continuous_feat = self.norm1(continuous_feat.transpose(0, 1) + continuous_out)
        
        # 门控融合
        combined = torch.cat([discrete_feat, continuous_feat], dim=-1)
        gate = self.gate(combined)  # 生成权重
        fused_feat = gate * discrete_feat + (1 - gate) * continuous_feat
        
        # 前馈网络
        fused_feat = self.norm2(fused_feat + self.ffn(fused_feat))
        
        return fused_feat

class TemporalConvNet(nn.Module):
    """时序卷积网络 (TCN)"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        
        # 残差连接
        if residual.shape[1] != out.shape[1]:
            residual = nn.Conv1d(residual.shape[1], out.shape[1], kernel_size=1).to(x.device)(residual)
        
        out = out + residual
        out = out.permute(0, 2, 1)  # (B, T, C)
        out = self.layer_norm(out)
        return out
    