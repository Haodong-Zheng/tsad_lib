import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs import Config
from utils.filter import filter

class Inception_Block_V1(nn.Module):
    """Inception块（多尺度卷积取平均）"""
    def __init__(self, in_channels, out_channels, num_kernels=Config.num_kernels, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []  # 多尺度卷积核
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=2*i+1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)  # 多尺度卷积结果取平均
        return res

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        
        # 双层Inception结构（含GELU激活）
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        #  全局top - k频率（所有样本相同周期）
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)  # # 两个Inception块+GELU
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation  振幅权重处理（全局softmax）
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)  # 加权和
        # residual connection
        res = res + x
        return res

class TimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, d_model, 
                 d_ff, top_k, num_kernels, e_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        if(Config.dataset == "SDC"):
            _, _, self.actual_enc_in = get_selected_indices(self)
            self.enc_in = self.actual_enc_in
        # 嵌入层（位置嵌入+值嵌入）
        self.enc_embedding = DataEmbedding(self.enc_in, d_model, dropout)
        
        # TimesBlock堆叠
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, enc_in)
        self.e_layers = e_layers

    def forward(self, x):
        # 序列归一化（预处理的数据已经归一化，无需归一化）
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev
        # print('x.shape:', x.shape)

        # 1. 筛选特征
        if(Config.dataset == "SDC"):
            discrete_indices, continuous_indices,_ = get_selected_indices(self)
            x = x[:, :, discrete_indices + continuous_indices]
        # 嵌入层
        enc_out = self.enc_embedding(x, None)  # [B, T, d_model]
        
        # TimesBlock处理
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # 投影层
        dec_out = self.projection(enc_out)
        # De-Normalization from Non-stationary Transformer

        # 反归一化（预处理的数据已经归一化，无需反归一化）
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out

# ============= 嵌入层实现 =============
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=Config.dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 排除0频率
    # 获取top-k频率索引
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    # 计算周期 p = T // index
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def get_selected_indices(self):
    discrete_indices, continuous_indices = filter()
    return discrete_indices, continuous_indices, len(discrete_indices + continuous_indices)
