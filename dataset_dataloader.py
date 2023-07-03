import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
from utils import mat2ten, time_features
import pandas as pd


class Get_Dataset(Dataset):
    def __init__(
            self, missing_pattern='RSM', 
            missing_rate=0.1,
            dataset_name="", 
            save_folder="", 
            seq_len=36,
            timeenc=0,
            is_train=True,
            timeenc_freq='h'
                ):
        """
        K: number of features, number of nodes
        D: number of days
        L_d: number of time intervals in a day
        """
        self.spatial_inp = None
        self.seq_len = seq_len
        self.timeenc = timeenc
        self.is_train = is_train

        if dataset_name == "PeMS7_V_228":
            path = "./dataset/PeMS7/" + "PeMSD7_V_228.csv"
            data_arr = pd.read_csv(path, header=None).values # (D*L_d, K)

            df_stamp = []
            for day in pd.date_range(start='2012-05-01', end='2012-06-30', freq='B'):
                date_index = pd.date_range(start=day, end=day + pd.offsets.Day(), freq='5min')[:-1]
                df_stamp.append(pd.Series(date_index))
            df_stamp = pd.concat(df_stamp)
            df_stamp = pd.DataFrame(df_stamp).rename(columns={0:'date'})

            path = "./dataset/PeMS7/" + "PeMSD7_W_228.csv"
            weight_A = pd.read_csv(path, header=None).values # (K, K) weighted adjacency matrix
            weight_A_norm = (weight_A - weight_A.mean()) / weight_A.std()
            self.spatial_inp = weight_A_norm
            # Select weekdays and exclude weekends

            L_d = 288 # interval 5 min

        elif dataset_name == "PeMS7_V_1026":
            path = "./dataset/PeMS7/" + "PeMSD7_V_1026.csv"
            data_arr = pd.read_csv(path, header=None).values # (D*L_d, K)
            date_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='D')
            
            path = "./dataset/PeMS7/" + "PeMSD7_W_1026.csv"
            weight_A = pd.read_csv(path, header=None).values # (K, K) weighted adjacency matrix
            weight_A_norm = (weight_A - weight_A.mean()) / weight_A.std()
            self.spatial_inp = weight_A_norm
            # Select weekdays and exclude weekends
            weekdays = date_range[date_range.weekday < 5].dayofweek
            dow_arr = weekdays.to_numpy() # 44 weekdays
            L_d = 288 # interval 5 min

            data_mat = np.reshape(data_arr, (len(dow_arr), L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)
            
        elif dataset_name == "Hangzhou":
            path = "./dataset/" + dataset_name + "/tensor.mat"
            data_mat = loadmat(path)['tensor'].transpose(0,2,1)  # of shape (K, L_d, D) ndarray
            dow_arr = self._generate_dow_array('2019/01/01', '2019/01/25')[0]

        elif dataset_name == "Seattle":
            data_arr_df = pd.read_pickle('./dataset/Seattle/speed_matrix_2015') # (D*L_d, K)
            # data_arr.to_csv('./dataset/Seattle/speed_matrix_2015.csv', index=True)
            location_info = pd.read_csv('./dataset/Seattle/Cabinet Location Information.csv')
            dow_arr, date_range = self._generate_dow_array('2015/01/01', '2015/12/31')
            D = len(date_range)
            L_d = 288

            # the sensor id's in data_arr_df actually has duplicated sensor ids
            distance_df = pd.DataFrame({'SensorName': [col[1:] for col in data_arr_df.columns]})
            dist_merged_df = pd.merge(distance_df, location_info[['CabName', 'Lat', 'Lon']], left_on='SensorName', right_on='CabName', how='left')

            # drop the redundant 'CabName' column
            dist_merged_df = dist_merged_df.drop('CabName', axis=1)

            # Compute the adjacency matrix
            adj_matrix = np.zeros((len(dist_merged_df), len(dist_merged_df)))
            for i in range(len(dist_merged_df)):
                for j in range(i+1, len(dist_merged_df)):
                    lat1, lon1 = dist_merged_df.iloc[i]['Lat'], dist_merged_df.iloc[i]['Lon']
                    lat2, lon2 = dist_merged_df.iloc[j]['Lat'], dist_merged_df.iloc[j]['Lon']
                    dist = self._haversine(lat1, lon1, lat2, lon2)
                    adj_matrix[i,j] = dist
                    adj_matrix[j,i] = dist

            weight_A_norm = (adj_matrix - adj_matrix.mean()) / adj_matrix.std()
            self.spatial_inp = weight_A_norm
            
            data_mat = np.reshape(data_arr_df.values, (D, L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)

        elif dataset_name == "Portland":
            path = "./dataset/" + dataset_name + "/volume.npy"
            data_arr = np.load(path)
            dim1, dim2 = data_arr.shape
            dim = np.array([dim1, 96, 31])
            data_mat = mat2ten(data_arr, dim, 0) # of shape (K, L_d, D) ndarray
            dow_arr = self._generate_dow_array('2021/01/01', '2021/01/31')[0]
        else:
            print(0)

        DL_d, dim_K = data_arr.shape

        num_train = int(int(DL_d * 0.9) - (int(DL_d * 0.9) % L_d))

        border1s = [0, num_train]
        border2s = [num_train, DL_d]

        if is_train:
            boarder1 = border1s[0]
            boarder2 = border2s[0]
            data_arr = data_arr[boarder1:boarder2]
        else:
            boarder1 = border1s[1]
            boarder2 = border2s[1]
            data_arr = data_arr[boarder1:boarder2]

        data_arr[np.isnan(data_arr)] = 0

        self.actual_mask = 1 - (data_arr == 0) # shape (D*L_d, K)
        
        if missing_pattern == 'RSM':
            np.random.seed(1000)
            selected_features = np.random.choice(dim_K, round(dim_K * missing_rate), replace=False)
            data_arr_missing = data_arr.copy()
            data_arr_missing[:, selected_features] = 0 # shape (D*L_d, K)

        elif missing_pattern == 'NRSM':
            np.random.seed(1000)
        else:
            # this missing pattern should be called mixed, but not usre if it's meaningful
            """
            I think this senario is still meaningful, in reality, if we wanna do network-wide estimation, 
            the observed data we use to infer the SM could be partially missing. we need to test the model
            performance/power under this senario.
            """
            data_mat_missing = data_arr

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
            data_stamp = data_stamp[boarder1:boarder2]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=timeenc_freq)
            data_stamp = data_stamp.transpose(1, 0) # from (time_emb_dim, L) to (L, time_emb_dim)
            data_stamp = data_stamp[boarder1:boarder2]

        self.missing_mask = 1 - (data_arr_missing == 0) # shape (D*L_d, K), 1 indicates observed, 0 indicates missing
        # save the missing mask for method comparison
        save_path = save_folder + '/' + 'tensor_missing.npz'
        np.savez(save_path, self.missing_mask)

        # both are of shape (K, )
        if self.is_train:
            self.training_mean, self.training_std = self._meanstd_calculator(data_arr, self.actual_mask) 
            path = save_folder + "/meanstd.pk"
            with open(path, "wb") as f:
                pickle.dump([self.training_mean, self.training_std], f)
        else:
            path = save_folder + "/meanstd.pk"
            with open(path, "wb") as f:
                self.training_mean, self.training_std = pickle.load(f)

        # normalization using mean and std of training data
        self.data_arr_norm = ((data_arr - self.training_mean) / self.training_std) * self.actual_mask

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = R * c
        return d

    def _meanstd_calculator(self, arr, mask):
        _, dim2 = arr.shape # arr's shape (D*L_d, K)
        mean = np.zeros(dim2)
        std = np.zeros(dim2)

        for k in range (dim2):
            k_arr = arr[:, k][mask[:, k] == 1]
            mean[k] = k_arr.mean()
            std[k] = k_arr.std()
        
        return mean, std
    
    def __len__(self):
        if self.is_train:
            return len(self.data_arr_norm) - self.seq_len + 1
        else:
            return int(len(self.data_arr_norm) / self.seq_len)

    def __getitem__(self, index):
        if self.is_train:
            s_begin = index
            s_end = s_begin + self.seq_len
        else:
            s_begin = index * self.seq_len
            s_end = s_begin + self.seq_len

        subseq = self.data_arr_norm[s_begin:s_end]
        missing_mask = self.missing_mask[s_begin:s_end]
        actual_mask = self.actual_mask[s_begin:s_end]

        if self.spatial_inp.any():
            spatial_inp = self.spatial_inp

        sample = {
            "actual_data": subseq,
            "missing_mask": missing_mask,
            "actual_mask": actual_mask,
            "timestamps": np.arange(subseq.shape[0]),
            "spatial_inp": spatial_inp
            }
        
        return sample

def get_dataloader(
        batch_size,
        device, 
        missing_pattern="RSM", 
        missing_rate=0.1, 
        dataset_name="", 
        save_folder="", 
        seq_length=144,
        timeenc = 1,
        is_train = True,
        timeenc_freq = 'h',
                ):
    
    dataset = Get_Dataset(
        missing_pattern, 
        missing_rate, 
        dataset_name, 
        save_folder, 
        seq_length,
        timeenc,
        is_train,
        timeenc_freq
        )
    
    if is_train:
        shuffle_flag = True
        drop_last = True
        batch_size = 1
    else:
        shuffle_flag = False
        drop_last = True

    tensor_mean = torch.from_numpy(dataset.training_mean).to(device).float()
    tensor_std = torch.from_numpy(dataset.training_std).to(device).float()

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )

    return data_loader, tensor_mean, tensor_std