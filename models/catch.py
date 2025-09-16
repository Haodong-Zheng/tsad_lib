import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from einops import rearrange, repeat
import math

class CATCH(nn.Module):
    def __init__(self, num_channels, seq_len, patch_size=8, patch_stride=4, 
                 hidden_dim=64, num_heads=4, num_layers=2, ffn_ratio=4,
                 lambda1=1.0, lambda2=0.1, lambda3=0.1, lambda_score=0.1, temperature=0.1):
        super(CATCH, self).__init__()
        
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_score = lambda_score
        self.temperature = temperature
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_size) // patch_stride + 1
        
        # Forward Module
        self.instance_norm = nn.InstanceNorm1d(num_channels, affine=True)
        
        # Projection layer for frequency patches
        self.projection = nn.Linear(2 * patch_size, hidden_dim)
        
        # Channel Fusion Module
        self.cfms = nn.ModuleList([
            CFM(num_channels, hidden_dim, num_heads, ffn_ratio, temperature)
            for _ in range(num_layers)
        ])
        
        # Time-Frequency Reconstruction Module
        # 计算展平后的维度
        self.flatten_dim = self.num_patches * hidden_dim
        
        # Separate projections for real and imaginary parts
        self.projection_r = nn.Linear(self.flatten_dim, seq_len)
        self.projection_i = nn.Linear(self.flatten_dim, seq_len)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_channels)
        batch_size, seq_len, num_channels = x.shape
        
        # 转置以适应InstanceNorm1d的要求 (需要通道在第二维)
        x_transposed = x.transpose(1, 2)  # (batch_size, num_channels, seq_len)
        
        # Forward Module
        # Instance normalization
        x_norm = self.instance_norm(x_transposed)  # (batch_size, num_channels, seq_len)
        
        # FFT and patching
        x_fft = fft.fft(x_norm, dim=-1)
        x_r, x_i = x_fft.real, x_fft.imag
        
        # Create patches for real and imaginary parts
        patches_r = x_r.unfold(-1, self.patch_size, self.patch_stride)  # (batch_size, num_channels, num_patches, patch_size)
        patches_i = x_i.unfold(-1, self.patch_size, self.patch_stride)
        
        # Concatenate real and imaginary parts
        patches = torch.cat([patches_r, patches_i], dim=-1)  # (batch_size, num_channels, num_patches, 2*patch_size)
        
        # Project to hidden dimension
        patches_proj = self.projection(patches)  # (batch_size, num_channels, num_patches, hidden_dim)
        
        # Channel Fusion Module
        # Process each patch separately
        patches_processed = patches_proj
        masks_list = []
        attention_scores_list = []
        masked_scores_list = []
        
        for cfm in self.cfms:
            patches_processed, masks, attention_scores, masked_scores = cfm(patches_processed)
            masks_list.extend(masks)
            attention_scores_list.extend(attention_scores)
            masked_scores_list.extend(masked_scores)
        
        # Separate real and imaginary parts for reconstruction
        patches_r_processed = patches_processed[:, :, :, :self.hidden_dim//2]
        patches_i_processed = patches_processed[:, :, :, self.hidden_dim//2:]
        
        # Time-Frequency Reconstruction Module
        # Flatten and reconstruct frequency domain
        patches_r_flat = rearrange(patches_r_processed, 'b c p h -> b c (p h)')
        patches_i_flat = rearrange(patches_i_processed, 'b c p h -> b c (p h)')
        
        # 确保展平后的维度正确
        assert patches_r_flat.shape[-1] == self.flatten_dim, f"Expected {self.flatten_dim}, got {patches_r_flat.shape[-1]}"
        
        x_r_recon = self.projection_r(patches_r_flat)  # (batch_size, num_channels, seq_len)
        x_i_recon = self.projection_i(patches_i_flat)  # (batch_size, num_channels, seq_len)
        
        # Reconstruct time domain
        x_fft_recon = torch.complex(x_r_recon, x_i_recon)
        x_recon = fft.ifft(x_fft_recon, dim=-1).real
        
        # 转置回原始形状 (batch_size, seq_len, num_channels)
        x_recon = x_recon.transpose(1, 2)
        x_r_recon = x_r_recon.transpose(1, 2)
        x_i_recon = x_i_recon.transpose(1, 2)
        
        # 转置原始FFT结果用于损失计算
        x_r = x_r.transpose(1, 2)
        x_i = x_i.transpose(1, 2)
        x_norm = x_norm.transpose(1, 2)
        
        return x_recon, x_r_recon, x_i_recon, masks_list, attention_scores_list, masked_scores_list
    
    def compute_loss(self, x, x_recon, x_r_recon, x_i_recon, x_r, x_i, masks, attention_scores, masked_scores):
        # Reconstruction losses
        rec_loss_time = torch.norm(x - x_recon, p=2) ** 2
        rec_loss_freq = torch.norm(x_r - x_r_recon, p=1) + torch.norm(x_i - x_i_recon, p=1)
        
        # Channel correlation discovering losses
        clustering_loss = 0
        regular_loss = 0
        
        for mask, attn, masked_attn in zip(masks, attention_scores, masked_scores):
            # Clustering loss (InfoNCE-like)
            n = mask.size(0)
            exp_masked = torch.exp(masked_attn / self.temperature)
            exp_original = torch.exp(attn / self.temperature)
            
            cluster_loss = -torch.log(exp_masked.sum(dim=-1) / exp_original.sum(dim=-1))
            clustering_loss += cluster_loss.mean()
            
            # Regularization loss
            identity = torch.eye(n, device=mask.device)
            regular_loss += torch.norm(identity - mask, p='fro') / n
        
        # Total loss
        total_loss = (rec_loss_time + 
                     self.lambda1 * rec_loss_freq + 
                     self.lambda2 * clustering_loss + 
                     self.lambda3 * regular_loss)
        
        return total_loss, rec_loss_time, rec_loss_freq, clustering_loss, regular_loss
    
    def anomaly_score(self, x, x_recon, x_r_recon, x_i_recon, x_r, x_i):
        # Time domain score
        time_score = torch.norm(x - x_recon, p=2, dim=-1) ** 2  # (batch_size, seq_len)
        
        # Frequency domain score
        freq_score = torch.norm(x_r - x_r_recon, p=1, dim=-1) + torch.norm(x_i - x_i_recon, p=1, dim=-1)
        
        # Combined score
        anomaly_score = time_score + self.lambda_score * freq_score
        
        return anomaly_score


class CFM(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_heads, ffn_ratio=4, temperature=0.1):
        super(CFM, self).__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        
        # Mask Generator
        self.mask_generator = MaskGenerator(num_channels, hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Attention projections
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_ratio * hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_ratio * hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, num_channels, num_patches, hidden_dim)
        batch_size, num_channels, num_patches, hidden_dim = x.shape
        
        # Process each patch independently
        outputs = []
        masks = []
        attention_scores_list = []
        masked_scores_list = []
        
        for i in range(num_patches):
            patch_x = x[:, :, i, :]  # (batch_size, num_channels, hidden_dim)
            
            # Generate mask for this patch
            mask_prob, mask = self.mask_generator(patch_x)  # (batch_size, num_channels, num_channels)
            
            # Layer normalization
            patch_x_norm = self.layer_norm(patch_x)
            
            # Compute queries, keys, values
            Q = self.w_q(patch_x_norm)  # (batch_size, num_channels, hidden_dim)
            K = self.w_k(patch_x_norm)  # (batch_size, num_channels, hidden_dim)
            V = self.w_v(patch_x_norm)  # (batch_size, num_channels, hidden_dim)
            
            # Reshape for multi-head attention
            Q = rearrange(Q, 'b c (h d) -> b h c d', h=self.num_heads)
            K = rearrange(K, 'b c (h d) -> b h c d', h=self.num_heads)
            V = rearrange(V, 'b c (h d) -> b h c d', h=self.num_heads)
            
            # Compute attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, num_channels, num_channels)
            
            # Apply mask
            mask_expanded = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (batch_size, num_heads, num_channels, num_channels)
            masked_attn_scores = attn_scores * mask_expanded + (1 - mask_expanded) * (-1e9)
            
            # Softmax
            attn_weights = torch.softmax(masked_attn_scores, dim=-1)  # (batch_size, num_heads, num_channels, num_channels)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, num_channels, head_dim)
            
            # Combine heads
            attn_output = rearrange(attn_output, 'b h c d -> b c (h d)')  # (batch_size, num_channels, hidden_dim)
            
            # Residual connection
            attn_output = attn_output + patch_x
            
            # Feed-forward network
            ff_output = self.ffn(attn_output)
            
            # Residual connection
            output = ff_output + attn_output
            
            outputs.append(output.unsqueeze(2))
            masks.append(mask)
            attention_scores_list.append(attn_scores.mean(dim=1))  # Average over heads
            masked_scores_list.append(masked_attn_scores.mean(dim=1))  # Average over heads
        
        # Combine patches
        output = torch.cat(outputs, dim=2)  # (batch_size, num_channels, num_patches, hidden_dim)
        
        return output, masks, attention_scores_list, masked_scores_list


class MaskGenerator(nn.Module):
    def __init__(self, num_channels, hidden_dim):
        super(MaskGenerator, self).__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Linear layers for mask generation
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_channels * num_channels)
        
        # Activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, num_channels, hidden_dim)
        batch_size, num_channels, hidden_dim = x.shape
        
        # Generate mask probabilities
        x_proj = self.linear1(x)
        x_proj = torch.relu(x_proj)
        mask_logits = self.linear2(x_proj)  # (batch_size, num_channels, num_channels * num_channels)
        
        # Reshape to get channel correlations
        mask_logits = mask_logits.view(batch_size, num_channels, num_channels, num_channels)
        
        # Average over the last dimension to get a matrix
        mask_logits = mask_logits.mean(dim=-1)  # (batch_size, num_channels, num_channels)
        
        # Apply sigmoid to get probabilities
        mask_prob = self.sigmoid(mask_logits)
        
        # Bernoulli sampling with Gumbel-Softmax reparameterization
        mask = self.gumbel_softmax(mask_prob, hard=True)
        
        # Ensure diagonal is always 1 (each channel is always relevant to itself)
        identity = torch.eye(num_channels, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        mask = torch.max(mask, identity)
        
        return mask_prob, mask
    
    def gumbel_softmax(self, logits, tau=1.0, hard=False, eps=1e-10):
        # Gumbel-Softmax reparameterization trick
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        y = (logits + gumbel_noise) / tau
        y = torch.sigmoid(y)
        
        if hard:
            # Straight-through estimator
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y
        
        return y
    