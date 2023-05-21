import numpy as np
import torch
import torch.nn as nn
from BiTrans_predictor import diff_CDI

class CDSTI_base(nn.Module):
    def __init__(self, spatial_dim, config, device):
        super().__init__()
        self.device = device
        self.spatial_dim = spatial_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_dow_dim = config["model"]["dowemb"]
        self.emb_tod_dim = config["model"]["todemb"]
        # + 1 for considering the conditional mask as one of the side info
        self.emb_side_dim = self.emb_feature_dim + 1
        self.emd_extra_tem_dim = self.emb_time_dim + self.emb_dow_dim + self.emb_tod_dim

        # feature dimension
        self.embed_layer = nn.Embedding(
            num_embeddings=self.spatial_dim, embedding_dim=self.emb_feature_dim
        )
        
        self.dow_emb_layer = nn.Embedding(
            num_embeddings=7, embedding_dim=self.emb_dow_dim
        )

        self.tod_emb_layer = nn.Embedding(
            num_embeddings=config['model']['toddim'], embedding_dim=self.emb_tod_dim
        )

        # parameters for diffusion models
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_side_dim
        config_diff["extra_temporal_dim"] = self.emd_extra_tem_dim

        # input_dim of the noise predictor
        # if conditional, then the one input sample is (x_{0}^{co}, x_{t}^{ta}) of shape (K,L,2) 
        input_dim = 2
        self.diffmodel = diff_CDI(config_diff, input_dim)

        # the num of steps for diffusion process, i.e., T
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        # self.beta is of shape (T,)
        self.alpha_hat = 1 - self.beta
        """
        self.alpha: cumulative product of alpha_hat, e.g., alpha[i] = alpha_hat[0] * alpha_hat[1] * ... * alpha_hat[i]
        """
        self.alpha = np.cumprod(self.alpha_hat) # self.alpha is still of shape (T,)
        # reshape for computing, self.alpha_torch is of shape (T,) -> (T,1,1)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        """
        sinusoidal position embedding for time embedding / timestamp embedding, not diffusion step embeeding
        pos is the tensor of timestamps, of shape (B, L)
        """
        # pe is of shape (B, L, emb_time_dim), where emb_time_dim = d_model
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2) # (B, L) -> (B, L, 1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    

    def get_side_info(self, cond_mask):
        B, K, L = cond_mask.shape

        feature_embed = self.embed_layer(
            torch.arange(self.spatial_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) # (K,emb) -> (1,1,K,emb) -> (B,L,K,feature_emb)
        feature_embed = feature_embed.permute(0,3,2,1) # (B,L,K,feature_emb) -> (B,feature_emb,K,L)

        side_mask = cond_mask.unsqueeze(1)  # (B,K,L) -> (B,1,K,L)
        side_info = torch.cat([feature_embed, side_mask], dim=1) # -> (B,feature_emb+1,K,L)

        return side_info
    
    def extra_temporal_feature(self, observed_tp, conda_mask, dow_arr, tod_arr):
        B, K, L = conda_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,emd) -> (B,L,1,emb) -> (B,L,K,time_emb)

        dow_embed = self.dow_emb_layer(dow_arr) # (B,L,emb)
        dow_embed = dow_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,emb) -> (B,L,1,emb) -> (B,L,K,dow_emb)

        tod_embed = self.tod_emb_layer(tod_arr) # (B,L,emb)
        tod_embed = tod_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,emb) -> (B,L,1,emb) -> (B,L,K,tod_emb)

        extra_info = torch.cat([time_embed, dow_embed, tod_embed], dim=-1)  # (B,L,K,*)
        extra_info = extra_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return extra_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, extra_tem_feature, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, extra_tem_feature, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, missing_mask, actual_mask, side_info, extra_tem_feature, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            # in Pytorch, shape of a tensor is specified using a list or tuple even if it is one-dimensional,
            # thus we use [B] instead of B
            # this randomly samples a seq of integers from [0, num_steps)
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        # alpha_torch is of shape (T,1,1), t is of torch.Size([B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data) # (B,K,L)
        noisy_data = (current_alpha ** 0.5) * observed_data + ((1.0 - current_alpha) ** 0.5) * noise

        target_mask = actual_mask - missing_mask
        # if unconditional, total_input = x_{t} = [x_{t}^{ta}, x_{t}^{co}], shape (B,1,K,L)
        # if conditional, total_input = [x_{t}^{ta}, cond_obs], cond_obs is x_{0}^{co}, shape (B,2,K,L)
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, missing_mask)

        # predicted noise
        predicted = self.diffmodel(total_input, side_info, extra_tem_feature, t)  # (B,K,L)

        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, missing_mask):

        cond_obs = (missing_mask * observed_data).unsqueeze(1) # (B,K,L) -> (B,1,K,L)
        noisy_target = ((1 - missing_mask) * noisy_data).unsqueeze(1) # (B,K,L) -> (B,1,K,L)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, extra_tem_feature, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):

                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, extra_tem_feature, torch.tensor([t]).to(self.device))

                coeff1 = 1 / (self.alpha_hat[t] ** 0.5)
                coeff2 = (1 - self.alpha_hat[t]) / ((1 - self.alpha[t]) ** 0.5)
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            actual_data, # x_0 (B,K,L)
            actual_mask, # (B,K,L)
            missing_mask, # (B,K,L)
            timestamps, # (B,L)
            dow_arr, # (B,L)
            tod_arr, # (B,L)
        ) = self.process_data(batch)


        side_info = self.get_side_info(missing_mask)
        extra_feature = self.extra_temporal_feature(timestamps, missing_mask, dow_arr, tod_arr)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(actual_data, missing_mask, actual_mask, side_info, extra_feature, is_train)

    def evaluate(self, batch, n_samples):
        (
            actual_data,
            actual_mask,
            missing_mask,
            timestamps,
            dow_arr,
            tod_arr,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = missing_mask
            observed_mask = actual_mask
            observed_tp = timestamps
            observed_data = actual_data

            target_mask = observed_mask - cond_mask
            extra_feature = self.extra_temporal_feature(timestamps, missing_mask, dow_arr, tod_arr)

            side_info = self.get_side_info(cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, extra_feature, n_samples)

            # for i in range(len(cut_length)):  # to avoid double evaluation
            #     target_mask[i, ..., 0 : cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp

class CDSTI(CDSTI_base):
    def __init__(self, config, device, spatial_dim):
        super(CDSTI, self).__init__(spatial_dim, config, device)
        
    def process_data(self, batch):
        actual_data = batch["actual_data"].to(self.device).float()
        actual_mask = batch["actual_mask"].to(self.device).float()
        missing_mask = batch["missing_mask"].to(self.device).float()
        timestamps = batch["timestamps"].to(self.device).float()
        dow_arr = batch["dow_arr"].to(self.device).long()
        tod_arr = batch["tod_arr"].to(self.device).long()

        actual_data = actual_data.permute(0, 2, 1)
        actual_mask = actual_mask.permute(0, 2, 1)
        missing_mask = missing_mask.permute(0, 2, 1)
        
        return (actual_data, actual_mask, missing_mask, timestamps, dow_arr, tod_arr)