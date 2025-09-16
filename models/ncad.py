# -*- coding: utf-8 -*-
"""
Modified Neural Contextual Anomaly Detection for Time Series (NCAD)
Core Model Only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class NCAD(nn.Module):
    
    def __init__(self, n_features, seq_len, rep_dim, hidden_dims, 
                 kernel_size, dropout, activation):
        super(NCAD, self).__init__()
        
        if isinstance(hidden_dims, str):
            hidden_dims = [int(x) for x in hidden_dims.split(',')]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        self.seq_len = seq_len
        self.n_features = n_features
        self.suspect_win_len = 5
        
        self.network = TCNnet(
            n_features=n_features,
            n_hidden=hidden_dims,
            n_output=n_features,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation
        )
        
        self.context_net = TCNnet(
            n_features=n_features,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation
        )
        
        self.final_net = TCNnet(
            n_features=n_features,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(rep_dim * 2, rep_dim),
            self._get_activation(activation),
            nn.Linear(rep_dim, n_features)
        )

    def _get_activation(self, activation):
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, x):
        # Input shape: (batch, seq, features)
        x_in = x.transpose(1, 2)  # Convert to (batch, features, seq)
        
        # Full reconstruction
        recon = self.network(x_in)
        
        # Context window
        context_end = self.seq_len - self.suspect_win_len
        context = x_in[:, :, :context_end]  # Context part
        
        # Feature extraction
        context_features = self.context_net(context)
        full_features = self.final_net(x_in)
        
        # Align time dimensions using padding
        pad_size = self.seq_len - context_features.shape[2]
        context_features = F.pad(context_features, (0, pad_size), "constant", 0)
        
        # Combine features along channel dimension
        combined_features = torch.cat([full_features, context_features], dim=1)
        
        # Process combined features
        combined_features = combined_features.transpose(1, 2)  # (batch, seq, channels)
        combined_output = self.output_layer(combined_features)
        
        # Final reconstruction combines both paths
        final_recon = recon.transpose(1, 2) + combined_output
        
        return final_recon


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, activation='ReLU'):
        super(TemporalBlock, self).__init__()
        # First convolution layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.activation1 = self._get_activation(activation)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.activation2 = self._get_activation(activation)
        self.dropout2 = nn.Dropout(dropout)
        
        # Downsample if needed
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.activation1, self.dropout1,
            self.conv2, self.chomp2, self.activation2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.act_final = self._get_activation(activation)
        
    def _get_activation(self, activation):
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act_final(out + res)


class TCNnet(nn.Module):
    def __init__(self, n_features, n_hidden, n_output, kernel_size=5, 
                 dropout=0.2, activation='ReLU'):
        super(TCNnet, self).__init__()
        layers = []
        num_levels = len(n_hidden)
        
        # Create temporal blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_hidden[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, 
                dilation=dilation_size, padding=(kernel_size-1)*dilation_size, 
                dropout=dropout, activation=activation
            )]
        
        output_layer = nn.Conv1d(n_hidden[-1] if num_levels > 0 else n_features, 
                                 n_output, kernel_size=1)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    