import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from torch.utils.data import TensorDataset, DataLoader

class DIF(nn.Module):
    def __init__(self, n_features, n_hidden, n_output, seq_len, bias=False):
        super(DIF, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.seq_len = seq_len
        self.bias = bias
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=n_hidden,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=bias
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=n_hidden,
                out_channels=n_output,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=bias
            ),
            nn.ReLU()
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=n_output,
                out_channels=n_hidden,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=bias
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=n_hidden,
                out_channels=n_features,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=bias
            )
        )
        
        # 自适应池化确保输出长度匹配
        self.pool = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x):
        # 原始输入形状: [batch, seq_len, features]
        # 转置为: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # 编码器处理
        x = self.encoder(x)
        
        # 确保长度正确
        x = self.pool(x)
        
        # 解码器重建
        x = self.decoder(x)
        
        # 转置回原始形状: [batch, seq_len, features]
        x = x.transpose(1, 2)
        return x

class DeepIsolationForestTS(nn.Module):
    """
    Deep Isolation Forest Model
    """
    def __init__(self, n_features, seq_len,
                 rep_dim=128, hidden_dims=128,
                 n_ensemble=10, n_estimators=5, 
                 max_samples=256):
        super(DeepIsolationForestTS, self).__init__()
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seq_len = seq_len
        
        self.models = nn.ModuleList([
            DIF(
                n_features=n_features,
                n_hidden=hidden_dims,
                n_output=rep_dim,
                seq_len=seq_len,
                bias=False
            ) for _ in range(n_ensemble)
        ])
        
        self.iForests = []

    def forward(self, x):
        idx = np.random.randint(0, self.n_ensemble)
        return self.models[idx](x)

    def fit_forest(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        dataset = TensorDataset(X)
        
        for i, model in enumerate(self.models):
            with torch.no_grad():
                features = self._extract_features(dataset, model)
            
            iforest = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=i
            )
            iforest.fit(features)
            self.iForests.append(iforest)

    def predict_anomaly(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        dataset = TensorDataset(X)
        
        scores = np.zeros((self.n_ensemble, len(X)))
        
        for i, (model, iforest) in enumerate(zip(self.models, self.iForests)):
            with torch.no_grad():
                features = self._extract_features(dataset, model)
            
            scores[i] = iforest.score_samples(features)
        
        return -np.mean(scores, axis=0)

    def _extract_features(self, dataset, model):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        features = []
        with torch.no_grad():
            for batch in dataloader:
                batch_x = batch[0]
                
                batch_x = batch_x.transpose(1, 2)
                feat = model.encoder(batch_x)
                
                feat = torch.mean(feat, dim=2).cpu().numpy()
                features.append(feat)
        
        return np.concatenate(features, axis=0)
    