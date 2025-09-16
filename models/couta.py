import torch
import numpy as np
from torch import nn
from numpy.random import RandomState

class COUTA(nn.Module):
 
    def __init__(self, n_features, seq_len, hidden_dims, 
                 kernel_size, dropout, bias):
        super(COUTA, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        # Process hidden dims configuration
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif isinstance(hidden_dims, str):
            hidden_dims = [int(dim) for dim in hidden_dims.split(',')]
        
        self.layers = nn.ModuleList()
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            in_channels = n_features if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            
            self.layers.append(
                TcnResidualBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation,
                    padding=padding, dropout=dropout,
                    bias=bias
                )
            )
        
        self.recon_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[-1], n_features * seq_len),
            nn.Unflatten(1, (seq_len, n_features))  # Reshape to original input shape
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, n_features)
        # TCN expects (batch_size, channels, seq_len)
        out = x.transpose(1, 2)  # (batch_size, n_features, seq_len)
        
        # Pass through TCN layers
        for layer in self.layers:
            out = layer(out)
            
        # Get last timestep features
        out = out[:, :, -1]  # (batch_size, hidden_dim)
        
        # Reconstruct original input
        reconstruction = self.recon_layers(out)
        
        return reconstruction  # (batch_size, seq_len, n_features)

class TcnResidualBlock(nn.Module):
    """Temporal Convolutional Network Residual Block"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation, padding, dropout=0.2, bias=True):
        super(TcnResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=bias
        )
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pad = nn.ConstantPad1d((padding, 0), 0)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=bias
        )
        
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=bias) \
            if in_channels != out_channels else None
            
    def forward(self, x):
        residual = x
        
        out = self.pad(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out += residual
        out = self.relu(out)
        return out
    