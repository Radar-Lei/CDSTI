import numpy as np
import torch
import torch.nn as nn
from ST_predictor import diff_CDI

class CDSTI_base(nn.Module):
    def __init__(self, spatial_dim, config, device):
        super().__init__()
        self.device = device
        self.spatial_dim = spatial_dim

        self.config = config

        self.emb_feature_dim = config["model"]["featureemb"]

        self.sampling_shrink_interval = config["model"]["sampling_shrink_interval"]

        # # feature dimension
        # self.embed_layer = nn.Embedding(
        #     num_embeddings=self.spatial_dim, embedding_dim=self.emb_feature_dim
        # )
        
        # input_dim of the noise predictor
        # if conditional, then the one input sample is (x_{0}^{co}, x_{t}^{ta}) of shape (K,L,2) 

        self.diffmodel = diff_CDI(config)

        # the num of steps for diffusion process, i.e., T
        self.num_steps = config['diffusion']["num_steps"]
        if config['diffusion']["schedule"] == "quad":
            self.beta = np.linspace(
                config['diffusion']["beta_start"] ** 0.5, config['diffusion']["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config['diffusion']["schedule"] == "linear":
            self.beta = np.linspace(
                config['diffusion']["beta_start"], config['diffusion']["beta_end"], self.num_steps
            )
        # self.beta is of shape (T,)
        self.alpha = 1 - self.beta
        """
        self.alpha_hat: cumulative product of alpha, e.g., alpha_hat[i] = alpha[0] * alpha[1] * ... * alpha[i]
        """
        self.alpha_hat = np.cumprod(self.alpha) # self.alpha is still of shape (T,)
        # reshape for computing, self.alpha_torch is of shape (T,) -> (T,1,1)
        self.alpha_torch = torch.tensor(self.alpha_hat).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, timeenc=128):
        """
        sinusoidal position encoding for time embedding / timestamp embedding, not diffusion step embeeding
        pos is the tensor of timestamps, of shape (B, L)
        """
        # pe is of shape (B, L, timeenc)
        pe = torch.zeros(pos.shape[0], pos.shape[1], timeenc).to(self.device)
        position = pos.unsqueeze(2) # (B, L) -> (B, L, 1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, timeenc, 2).to(self.device) / timeenc
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    

    def calc_loss(
        self, observed_data, missing_mask, actual_mask, seq_stamps):
        B, K, L = observed_data.shape

        t = torch.randint(0, self.num_steps, [B]).to(self.device)

        # alpha_torch is of shape (T,1,1), t is of torch.Size([B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data) # (B,K,L)
        noisy_data = (current_alpha ** 0.5) * observed_data + ((1.0 - current_alpha) ** 0.5) * noise

        target_mask = actual_mask - missing_mask
        # if conditional, total_input = [x_{t}^{ta}, cond_obs], cond_obs is x_{0}^{co}, shape (B,2,K,L)
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, missing_mask)

        # predicted noise
        predicted = self.diffmodel(total_input, t, seq_stamps)  # (B,K,L)

        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, missing_mask):

        cond_obs = (missing_mask * observed_data).unsqueeze(1) # (B,K,L) -> (B,1,K,L)
        noisy_target = ((1 - missing_mask) * noisy_data).unsqueeze(1) # (B,K,L) -> (B,1,K,L)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, seq_stamps, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):

            sample = torch.randn_like(observed_data)

            # if shring interval = -2, then 99, 97, 95, ... -1, 50 reverse steps
            # if shring interval = -1, then 99, 98, 97, there's no shrink
            s = self.num_steps - 1
            while True:
                if s < self.sampling_shrink_interval:
                    break
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1-cond_mask) * sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, torch.tensor([s]).to(self.device), seq_stamps)

                coeff = self.alpha_hat[s-self.sampling_shrink_interval]

                sigma = (((1 - coeff) / (1 - self.alpha_hat[s])) ** 0.5) * ((1-self.alpha_hat[s] / coeff) ** 0.5)

                sample = (coeff ** 0.5) * ((sample - ((1-self.alpha_hat[s]) ** 0.5) * predicted) / (self.alpha_hat[s] ** 0.5)) + ((1 - coeff - sigma ** 2) ** 0.5) * predicted

                # if s == self.sampling_shrink_interval, then it's the last step, no need to add noise
                if s > self.sampling_shrink_interval: 
                    noise = torch.randn_like(sample)
                    sample += sigma * noise

                s -= self.sampling_shrink_interval

            imputed_samples[:, i] = sample.detach()
        return imputed_samples
    
    def sm_mask_generator(self, actual_mask, missing_rate, missing_pattern='RSM'):
        """
        generate the missing mask for SM missing pattern,
        should follow the same strategy as dataloader to 
        select cols to be structurally missing, but without
        fixed random seed for training set diversity.

        return: (B,K,L) as the cond_mask in model training
        """
        # actual_mask: (B,K,L)
        copy_mask = actual_mask.clone()
        if missing_pattern == 'RSM':
            all_zero_indices = np.unique(torch.all(actual_mask == 0, dim=2).nonzero()[:, 1].cpu().numpy())
            _, dim_K, _ = copy_mask.shape
            available_features = [i for i in range(dim_K) if i not in all_zero_indices]
            selected_features = np.random.choice(available_features, round(len(available_features) * missing_rate), replace=False)
            copy_mask[:, selected_features, :] = 0
        elif missing_pattern == 'RM':
            nested_mask = np.random.choice([True, False], size=copy_mask.shape, p=[missing_rate, 1-missing_rate])
            copy_mask[nested_mask] = 0
            
        return copy_mask
        

    def forward(self, batch):
        (
            actual_data, # x_0 (B,K,L)
            _, # (B,K,L)
            actual_mask, # missing_mask as the actual mask
            seq_stamps, # (B,L, timeenc)
            _,
            spa_mat, # (B,K,K)
        ) = self.process_data(batch)
        

        missing_mask = self.sm_mask_generator(actual_mask, self.config["model"]["missing_rate"], self.config["model"]["missing_pattern"])
        # extra spatial feature shape: (B, K, K, L)  # (B,K,K) -> (B,K,K,1) -> (B,K,K,L)
        # extra_spa_feature = spa_mat.unsqueeze(-1).expand(-1, -1, -1, actual_data.shape[-1])
        loss_func = self.calc_loss

        return loss_func(actual_data, missing_mask, actual_mask, seq_stamps)

    def evaluate(self, batch, n_samples):
        (
            actual_data,
            actual_mask,
            missing_mask,
            seq_stamps,
            timestamps,
            spa_mat, # (B,K,K)
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = missing_mask
            observed_mask = actual_mask
            observed_tp = timestamps
            # to keep the ground truth unchanged
            observed_data = actual_data.clone()

            target_mask = observed_mask - cond_mask

            samples = self.impute(observed_data, cond_mask, seq_stamps, n_samples)

            # for i in range(len(cut_length)):  # to avoid double evaluation
            #     target_mask[i, ..., 0 : cut_length[i].item()] = 0

        return samples, actual_data, target_mask, observed_mask, observed_tp

class CDSTI(CDSTI_base):
    def __init__(self, config, device, spatial_dim):
        super(CDSTI, self).__init__(spatial_dim, config, device)
        
    def process_data(self, batch):
        actual_data = batch["actual_data"].to(self.device).float()
        actual_mask = batch["actual_mask"].to(self.device).float()
        missing_mask = batch["missing_mask"].to(self.device).float()
        seq_stamps = batch["seq_stamps"].to(self.device).float()
        timestamps = batch["timestamps"].to(self.device).float()
        spatial_mat = batch["spatial_inp"].to(self.device).float() # (K,K)

        # (B.L,K) to (B,K,L)
        actual_data = actual_data.permute(0, 2, 1) 
        actual_mask = actual_mask.permute(0, 2, 1)
        missing_mask = missing_mask.permute(0, 2, 1)
        
        return (actual_data, actual_mask, missing_mask, seq_stamps, timestamps, spatial_mat)