import numpy as np
import torch
import torch.nn as nn
from submodules import DataEmbedding, Inception_Block_V1
import torch.fft
import torch.nn.functional as F

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=32, dropout=0.2, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model * 2, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model * 2,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
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
            B, C, num_period, period = out.shape
            # out = out.squeeze(2) # (B, C, L)
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        
        if configs.diff_schedule == "quad":
            self.beta = np.linspace(
                configs.beta_start ** 0.5, configs.beta_end ** 0.5, configs.diff_steps
            ) ** 2
        elif configs.diff_schedule == "linear":
            self.beta = np.linspace(
                configs.beta_start, configs.beta_end, configs.diff_steps
            )
        # self.beta is of shape (T,)
        self.alpha = 1 - self.beta
        """
        self.alpha_hat: cumulative product of alpha, e.g., alpha_hat[i] = alpha[0] * alpha[1] * ... * alpha[i]
        """
        self.alpha_hat = np.cumprod(self.alpha) # self.alpha is still of shape (T,)
        # reshape for computing, self.alpha_torch is of shape (T,) -> (T,1,1)
        self.alpha_torch = torch.tensor(self.alpha_hat).float().to(configs.gpu).unsqueeze(1).unsqueeze(1)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, configs.diff_steps)

        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model * 2)        

        self.projection = nn.Linear(configs.d_model * 2, configs.c_out, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, target_mask=None):
        B,L,K = x_enc.shape

        t = torch.randint(0, self.configs.diff_steps, [B]).to(self.configs.gpu)

        # alpha_torch is of shape (T,1,1), t is of torch.Size([B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(x_enc) # (B,L,K)
        noisy_data = (current_alpha ** 0.5) * x_enc + ((1.0 - current_alpha) ** 0.5) * noise

        cond_obs = mask * x_enc
        noisy_target = target_mask * noisy_data
        
        # embedding # enc_out is of shape (B, L_hist, 2*d_model)
        # also embedding diffusion step t of shape ([B])
        enc_out = self.enc_embedding(cond_obs, noisy_target, x_mark_enc, t)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # dec_out is of shape (B, L_pred, K)
        dec_out = self.projection(enc_out)

        return dec_out, noise
    
    def evaluate_acc(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, target_mask=None):
        B,L,K = x_enc.shape
        imputed_samples = torch.zeros(B, self.configs.diff_samples, L, K).to(self.configs.gpu)

        for i in range(self.configs.diff_samples):
            # diffusion starts from a pure Gaussian noise
            sample = torch.randn_like(x_enc)

            # initial diffusion step start from N-1
            # if shring interval = -2, then 99, 97, 95, ... -1, 50 reverse steps
            # if shring interval = -1, then 99, 98, 97, there's no shrink
            s = self.configs.diff_steps - 1

            while True:
                if s < self.configs.sampling_shrink_interval:
                    break
                
                cond_obs = mask * x_enc
                noisy_target = target_mask * sample

                # embedding # enc_out is of shape (B, L_hist, 2*d_model)
                enc_out = self.enc_embedding(cond_obs, noisy_target, x_mark_enc, torch.tensor([s]).to(self.configs.gpu))

                for j in range(self.layer):
                    enc_out = self.layer_norm(self.model[j](enc_out))

                # dec_out is of shape (B, L_pred, K)
                dec_out = self.projection(enc_out)

                coeff = self.alpha_hat[s-self.configs.sampling_shrink_interval]
                sigma = (((1 - coeff) / (1 - self.alpha_hat[s])) ** 0.5) * ((1-self.alpha_hat[s] / coeff) ** 0.5)
                sample = (coeff ** 0.5) * ((sample - ((1-self.alpha_hat[s]) ** 0.5) * dec_out) / (self.alpha_hat[s] ** 0.5)) + ((1 - coeff - sigma ** 2) ** 0.5) * dec_out

                # if s == self.sampling_shrink_interval, then it's the last step, no need to add noise
                if s > self.configs.sampling_shrink_interval: 
                    noise = torch.randn_like(sample)
                    sample += sigma * noise

                s -= self.configs.sampling_shrink_interval

            imputed_samples[:, i] = sample.detach()

        return imputed_samples
    
    def evaluate(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, target_mask=None):
        B,L,K = x_enc.shape
        imputed_samples = torch.zeros(B, self.configs.diff_samples, L, K).to(self.configs.gpu)

        for i in range(self.configs.diff_samples):
            # diffusion starts from a pure Gaussian noise
            sample = torch.randn_like(x_enc)

            # initial diffusion step start from N-1
            for s in range(self.configs.diff_steps - 1, -1, -1):
                cond_obs = mask * x_enc
                noisy_target = target_mask * sample

                # embedding # enc_out is of shape (B, L_hist, 2*d_model)
                enc_out = self.enc_embedding(cond_obs, noisy_target, x_mark_enc, torch.tensor([s]).to(self.configs.gpu))

                for j in range(self.layer):
                    enc_out = self.layer_norm(self.model[j](enc_out))

                # dec_out is of shape (B, L_pred, K)
                dec_out = self.projection(enc_out)

                coeff1 = 1 / self.alpha[s] ** 0.5
                coeff2 = (1 - self.alpha[s]) / (1 - self.alpha_hat[s]) ** 0.5

                sample = coeff1 * (sample - coeff2 * dec_out)

                # if then it's the last step, no need to add noise
                if s > 0: 
                    noise = torch.randn_like(sample)
                    sigma = (
                        (1.0 - self.alpha_hat[s - 1]) / (1.0 - self.alpha_hat[s]) * self.beta[s]
                    ) ** 0.5                    
                    sample += sigma * noise
            
            # use detech to create new tensor on the device, reserve sample for next iteration
            imputed_samples[:, i] = sample.detach()

        return imputed_samples