import torch
import torch.nn as nn
import torch.nn.functional as F

class FITS(nn.Module):
    """
    参数:
    ----------
    orig_seq_len : int
        原始序列长度（下采样前）
    DSR : int
        下采样率
    individual : bool
        是否为每个通道使用独立的上采样器
    enc_in : int
        输入特征数
    cut_freq : int
        截止频率，用于频率过滤
    """
    
    def __init__(self, orig_seq_len, DSR, individual, enc_in, cut_freq):
        super(FITS, self).__init__()
        self.orig_seq_len = orig_seq_len
        self.DSR = DSR
        self.individual = individual
        self.channels = enc_in
        self.down_len = (orig_seq_len + DSR - 1) // DSR
        
        # 计算下采样后的序列长度
        self.seq_len_down = (orig_seq_len + DSR - 1) // DSR
        
        # 频率相关参数
        self.length_ratio = orig_seq_len / self.seq_len_down
        
        # 计算下采样序列的FFT频率分量数量
        max_freq_bins = self.down_len // 2 + 1
        self.cut_freq = min(cut_freq, max_freq_bins)
        self.dominance_freq = self.cut_freq

        # 频率上采样器
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, 
                              int(self.dominance_freq * self.length_ratio),
                              dtype=torch.cfloat)
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, 
                int(self.dominance_freq * self.length_ratio),
                dtype=torch.cfloat
            )

    def forward(self, x):
        """前向传播
        
        输入:
            x: 已归一化的原始序列 [batch_size, orig_seq_len, n_features]
            
        输出:
            xy: 重建序列 [batch_size, orig_seq_len, n_features]
        """
        # 1. 下采样
        x_down = x[:, ::self.DSR, :]  # [batch_size, seq_len_down, n_features]
        
        # 2. 直接进行频率域处理
        low_specx = torch.fft.rfft(x_down, dim=1)  # [batch_size, seq_len_down//2+1, n_features]
        
        # 实际频率分量数量
        actual_freq_bins = low_specx.size(1)
        
        # 调整截止频率确保不越界
        cut_freq = min(self.cut_freq, actual_freq_bins)
        
        # 低通滤波 (保留截止频率前的分量)
        low_specx = low_specx[:, :cut_freq, :]
        
        # 3. 频率上采样
        if self.individual:
            # 对每个特征通道独立处理
            upsampled_freq_bins = int(cut_freq * self.length_ratio)
            low_specxy_ = torch.zeros(
                [x_down.size(0), upsampled_freq_bins, self.channels],
                dtype=torch.cfloat, device=x.device
            )
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[..., i])
        else:
            # 所有通道共享上采样器
            upsampled_freq_bins = int(self.cut_freq * self.length_ratio)
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. 构建完整频谱
        full_freq_bins = self.orig_seq_len // 2 + 1
        low_specxy = torch.zeros(
            [x_down.size(0), full_freq_bins, self.channels],
            dtype=torch.cfloat, device=x.device
        )
        copy_len = min(low_specxy_.shape[1], low_specxy.shape[1])
        low_specxy[:, :copy_len, :] = low_specxy_[:, :copy_len, :]
        
        # 5. 逆变换回时域
        low_xy = torch.fft.irfft(low_specxy, dim=1, n=self.orig_seq_len)
        low_xy = low_xy * self.length_ratio
        
        # 6. 直接输出
        return low_xy
    