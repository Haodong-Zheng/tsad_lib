import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class AnomalyTransformer(nn.Module):
    def __init__(self, seq_len, n_features, lr, epochs, batch_size,
                 epoch_steps, k, d_model, n_heads, e_layers):
        super(AnomalyTransformer, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.k = k
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        
        self.net = AnomalyTransformerModel(
            win_size=self.seq_len,
            enc_in=self.n_features,
            c_out=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers
        )
        
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)

    def forward(self, x):
        output, _, _, _ = self.net(x)
        return output

    def fit(self, X):
        self.train()
        for e in range(self.epochs):
            t1 = time.time()
            loss = self._train_epoch(X)
            print(f'Epoch {e+1:3d}/{self.epochs}, loss: {loss:.6f}, time: {time.time()-t1:.1f}s')
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            loss, _ = self._inference(X)
            return np.mean(loss, axis=1)

    def _train_epoch(self, X):
        criterion = nn.MSELoss()
        loss_list = []
        
        num_samples = X.shape[0]
        indices = torch.randperm(num_samples)
        
        for ii in range(0, num_samples, self.batch_size):
            if ii > self.epoch_steps * self.batch_size and self.epoch_steps != -1:
                break
                
            batch_idx = indices[ii:ii+self.batch_size]
            batch_x = X[batch_idx].float()
            
            self.optimizer.zero_grad()
            output, series, prior, _ = self.net(batch_x)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_len)
                series_loss += torch.mean(my_kl_loss(series[u], norm_prior.detach())) + \
                              torch.mean(my_kl_loss(norm_prior.detach(), series[u]))
                prior_loss += torch.mean(my_kl_loss(norm_prior, series[u].detach())) + \
                            torch.mean(my_kl_loss(series[u].detach(), norm_prior))
            
            series_loss /= len(prior)
            prior_loss /= len(prior)
            rec_loss = criterion(output, batch_x)

            loss1 = rec_loss - self.k * series_loss
            loss2 = rec_loss + self.k * prior_loss
            loss1.backward(retain_graph=True)
            loss2.backward()
            self.optimizer.step()

            loss_list.append(loss1.item())

        self.scheduler.step()
        return np.mean(loss_list)

    def _inference(self, X):
        criterion = nn.MSELoss(reduction='none')
        temperature = 50
        attens_energy = []
        
        num_samples = X.shape[0]
        for ii in range(0, num_samples, self.batch_size):
            batch_x = X[ii:ii+self.batch_size].float()
            output, series, prior, _ = self.net(batch_x)

            rec_loss = torch.mean(criterion(batch_x, output), dim=-1)
            
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_len)
                if u == 0:
                    series_loss = my_kl_loss(series[u], norm_prior.detach()) * temperature
                    prior_loss = my_kl_loss(norm_prior, series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], norm_prior.detach()) * temperature
                    prior_loss += my_kl_loss(norm_prior, series[u].detach()) * temperature
            
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * rec_loss
            attens_energy.append(cri.cpu().numpy())

        return np.concatenate(attens_energy, axis=0), None


class AnomalyTransformerModel(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff=64,
                 dropout=0.1, activation='gelu', output_attention=True):
        super(AnomalyTransformerModel, self).__init__()
        self.output_attention = output_attention

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, 
                                       output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=None 
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        return enc_out, series, prior, sigmas


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return x + y, attn, mask, sigma 


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list, prior_list, sigma_list = [], [], []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            
        return x, series_list, prior_list, sigma_list  # 跳过norm(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        self.register_buffer('distances', torch.zeros((win_size, win_size)))
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is None:
            attn_mask = TriangularCausalMask(B, L)
            scores.masked_fill_(attn_mask.mask.to(scores.device), -np.inf)
            
        attn = scale * scores
        sigma = sigma.transpose(1, 2)  # B L H -> B H L
        
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)
        
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return V.contiguous(), series, prior, sigma
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)
        
        sigma_input = queries.reshape(B, L, self.d_model)
        sigma = self.sigma_projection(sigma_input).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma, attn_mask
        )
        out = out.view(B, L, -1)
        
        return self.out_projection(out), series, prior, sigma
    