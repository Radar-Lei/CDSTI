import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

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
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        five_min_size = 288
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        elif freq == '5min':
            self.minute_embed = Embed(five_min_size, d_model)
        
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # see features_by_offsets in utils.py
        freq_map = {'h': 4, 't': 4, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, '5min': 5}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

def fea_encoding(pos, device, d_model=128):
    """
    sinusoidal position embedding for time embedding / timestamp embedding, not diffusion step embeeding
    pos is the tensor of timestamps, of shape (B, L)
    """
    # pe is of shape (B, L, emb_time_dim), where emb_time_dim = d_model
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device)
    position = pos.unsqueeze(2) # (B, L) -> (B, L, 1)
    div_term = 1 / torch.pow(
        10000.0, torch.arange(0, d_model, 2).to(device) / d_model
    )
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        # square bracket indexing 
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, dim, 1) / dim
        ) 
        div_term = div_term.unsqueeze(0)  # (1,dim)
        table = steps * div_term  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, diff_steps=100):
        super(DataEmbedding, self).__init__()

        self.cond_value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.noisy_value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.fea_reduce = nn.Linear(c_in, 1, bias=False)
        # embedding layer for diffusion step
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, cond_obs, noisy_target, x_mark, diff_step):
        B, L_hist, K = cond_obs.shape


        # fea_pos = np.arange(10).repeat(int(K/10))
        # fea_pos = torch.from_numpy(np.expand_dims(fea_pos, axis=0).repeat(B, axis=0)).to(cond_obs.device)
        # # fea_pos is of shape (B, K)
        # # fea_embedding is a tensor of shape (B, K, d_model)
        # fea_embedding = fea_encoding(fea_pos, cond_obs.device, d_model=self.d_model)
        # # (B, K, d_model) -> (B, 1, K, d_model) -> (B, L_hist, K, d_model) -> (B, L_hist, d_model, K)
        # fea_embedding = fea_embedding.unsqueeze(1).expand([B, L_hist, K, self.d_model]).permute(0,1,3,2)
        # # (B, L_hist, d_model, K) -> (B, L_hist, d_model, 1) -> (B, L_hist, d_model)
        # fea_embedding = self.fea_reduce(fea_embedding).squeeze(-1)


        # (B,) -> (B, d_model) -> (B, 1, d_model), this will broadcasting to (B, L_hist, d_model)
        diff_step_embedding = self.diffusion_embedding(diff_step).unsqueeze(1)

        cond_pos_embedding = self.position_embedding(cond_obs)
        noisy_pos_embedding = self.position_embedding(noisy_target)

        tem_embedding = self.temporal_embedding(x_mark)
        
        if x_mark is None:
            # self.value_embedding(x) is of shape (B, L_hist, d_model)
            # fea_embedding is of shape (B, L_hist, d_model)
            # self.position_embedding(x) is of shape (1, L_hist, d_model)
            x_cond = self.cond_value_embedding(cond_obs) + cond_pos_embedding + diff_step_embedding
            x_noisy = self.noisy_value_embedding(noisy_target) + noisy_pos_embedding + diff_step_embedding
        else:
            x_cond = self.cond_value_embedding(cond_obs) + tem_embedding + cond_pos_embedding + diff_step_embedding
            x_noisy = self.noisy_value_embedding(noisy_target) + tem_embedding + noisy_pos_embedding + diff_step_embedding
        
        # x is of shape (B, L_hist, 2*d_model)
        x = torch.cat([x_cond, x_noisy], dim=-1)

        return self.dropout(x)
    

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=4 * i + 1, padding=2*i))
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
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res