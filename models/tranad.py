import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder
import math

class TranAD(nn.Module):
    def __init__(self, feats, n_window=100):
        super(TranAD, self).__init__()
        self.n_feats = feats
        self.n_window = n_window
        self.d_model = 2 * feats
        
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, 
            dropout=0.2, 
            max_len=self.n_window
        )
        
        # Transformer编码器层（注意：nhead需是d_model的约数）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=feats if feats % 2 == 0 else 2, 
            dim_feedforward=512,
            dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=feats if feats % 2 == 0 else 2,
            dim_feedforward=512, 
            dropout=0.2
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        self.fcn = nn.Sequential(
            nn.Linear(self.d_model, feats),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        x_combined = torch.cat((x, c), dim=2)
        x_combined = x_combined * math.sqrt(self.d_model)
        x_combined = x_combined.transpose(0, 1)
        memory = self.pos_encoder(x_combined)
        memory = self.transformer_encoder(memory)
        
        tgt = x.transpose(0, 1)
        tgt = tgt.repeat(1, 1, 2)
        
        return tgt, memory

    def forward(self, x):

        c = torch.zeros_like(x)
        tgt, memory = self.encode(x, c)
        
        x1 = self.transformer_decoder1(tgt, memory)
        x1 = x1.transpose(0, 1)
        x1 = self.fcn(x1)
        
        c = (x1 - x) ** 2
        
        tgt, memory = self.encode(x, c)
        
        x2 = self.transformer_decoder2(tgt, memory)
        x2 = x2.transpose(0, 1)
        x2 = self.fcn(x2)
        
        return x2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    