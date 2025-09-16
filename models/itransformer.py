import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 直接处理转置后的输入: [B, D, L] -> [B, D, H]
        return self.dropout(self.value_embedding(x))

class EfficientAttention(nn.Module):
    def __init__(self, scale, attention_dropout=0.1):
        super(EfficientAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v):
        # q, k, v: [B, H, L, E]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        self.head_dim = d_keys

        self.qkv_projection = nn.Linear(d_model, 3 * n_heads * d_keys)
        self.out_projection = nn.Linear(n_heads * d_keys, d_model)
        
        self.scale = 1.0 / math.sqrt(d_keys)
        self.attention = EfficientAttention(scale=self.scale)

    def forward(self, x):
        B, D, H = x.shape  # [B, D, H]
        
        # 单次投影获取q,k,v
        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头形式 [B, D, H, E]
        q = q.view(B, D, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, D, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, D, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        out, attn = self.attention(q, k, v)
        
        # 合并多头输出
        out = out.permute(0, 2, 1, 3).contiguous().view(B, D, -1)
        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x, None

class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x, None

class iTransformer(nn.Module):
    """
    iTransformer模型
    输入: (batch_size, seq_len, n_features) -> [B, L, D]
    输出: (batch_size, seq_len, n_features) -> [B, L, D]
    """
    def __init__(self, enc_in, seq_len, d_model, n_heads, e_layers, d_ff, dropout=0.1):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        
        # 嵌入层
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        
        # 编码器
        self.encoder = Encoder(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        
        # 输出投影层
        self.projection = nn.Linear(d_model, seq_len)

    def forward(self, x):
        # 输入形状: [B, L, D] -> 转置为 [B, D, L]
        x = x.transpose(1, 2)
        
        # 嵌入层: [B, D, L] -> [B, D, H]
        enc_out = self.enc_embedding(x)
        
        # 编码器处理: [B, D, H] -> [B, D, H]
        enc_out, _ = self.encoder(enc_out)
        
        # 输出投影: [B, D, H] -> [B, D, L]
        dec_out = self.projection(enc_out)
        
        # 转置回原始维度: [B, D, L] -> [B, L, D]
        return dec_out.transpose(1, 2)
    