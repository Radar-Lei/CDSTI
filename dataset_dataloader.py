import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import pickle
from utils import mat2ten
import pandas as pd

class Get_Dataset(Dataset):
    def __init__(
            self, missing_pattern='RSM', 
            missing_rate=0.1,
            dataset_name="", 
            save_folder="", 
            seq_len=36,
            is_train=True,
                ):
        """
        K: number of features, number of nodes
        D: number of days
        L_d: number of time intervals in a day
        """
        self.spatial_inp = None
        self.seq_len = seq_len
        self.is_train = is_train
        self.missing_pattern = missing_pattern

        if dataset_name == "PeMS7_V_228":
            path = "./dataset/PeMS7/" + "PeMSD7_V_228.csv"
            data_arr = pd.read_csv(path, header=None).values # (D*L_d, K)
            date_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='D')
            
            path = "./dataset/PeMS7/" + "PeMSD7_W_228.csv"
            weight_A = pd.read_csv(path, header=None).values # (K, K) weighted adjacency matrix
            weight_A_norm = (weight_A - weight_A.mean()) / weight_A.std()
            self.spatial_inp = weight_A_norm
            # Select weekdays and exclude weekends
            weekdays = date_range[date_range.weekday < 5].dayofweek
            dow_arr = weekdays.to_numpy() # 44 weekdays
            L_d = 288 # interval 5 min

            data_mat = np.reshape(data_arr, (len(dow_arr), L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)
        elif dataset_name == "Competition_flow":
            path = "./dataset/competition/train-5min/" + "flow.csv"
            def my_date_parser(date_str):
                return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
            
            data_arr = pd.read_csv(path, index_col='date', parse_dates=True, date_parser=my_date_parser).values
            dow_arr, date_range = self._generate_dow_array('2023/04/02', '2023/06/30')
            D = len(date_range)
            L_d = 288
            num_locations = 10
            sensors_per_location = 4
            dist_list = [1260, 310, 900, 1010, 290, 2220, 240, 1220, 370]

            # Initialize a 20x20 adjacency matrix filled with zeros
            adj_matrix = np.zeros((40, 40))

            # Iterate through the rows of the matrix
            for i in range(40):
                # Determine the location index for the current row
                loc_i = i // sensors_per_location
                
                # Iterate through the columns of the matrix
                for j in range(40):
                    # Determine the location index for the current column
                    loc_j = j // sensors_per_location
                    
                    # Calculate the distance between the two locations
                    if loc_i == loc_j:
                        distance = 1
                    else:
                        min_loc = min(loc_i, loc_j)
                        max_loc = max(loc_i, loc_j)
                        distance = sum(dist_list[min_loc:max_loc])

                    # Set the adjacency matrix value at the current row and column
                    adj_matrix[i, j] = distance

            weight_A_norm = (adj_matrix - adj_matrix.mean()) / adj_matrix.std()
            self.spatial_inp = weight_A_norm

            data_mat = np.reshape(data_arr, (len(dow_arr), L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)

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

        dim_K, dim2_L_d, dim3_D = data_mat.shape

        self.dow_arr = np.repeat(dow_arr, dim2_L_d) # from (D, ) to (D*L_d, )
        self.tod_arr = np.concatenate([np.arange(dim2_L_d)] * dim3_D) # creating (L_d,) and then repeat D times to (D*L_d, )

        # reshap the data_arr to (D*L_d, K)
        # to maintain the order, first transpose (K, L_d, D) to (D, L_d, K), notice the feature dim need to place 
        # D first and then L_d since 
        # we want to the order as day first and then time points in a day, 
        data_arr = np.transpose(data_mat, (2, 1, 0)).reshape((data_mat.shape[2] * data_mat.shape[1], data_mat.shape[0]))

        DL_d, dim_K = data_arr.shape

        num_train = int(dim3_D * 0.9) * dim2_L_d

        self.test_day = dim3_D - int(dim3_D * 0.9)

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
        elif missing_pattern == 'RM':
            DL_d, dim_K = data_arr.shape
            np.random.seed(1000)
            num_zeros = int(DL_d * dim_K * 0.5)
            zero_indices = np.random.choice(DL_d * dim_K, num_zeros, replace=False)
            zero_indices = np.unravel_index(zero_indices, (DL_d, dim_K))
            data_arr_missing = data_arr.copy()
            data_arr_missing[zero_indices] = 0 # shape (D*L_d, K)
        elif missing_pattern == 'fixed':
            self.missing_mask = self.create_fixed_mask(dim_K)
        elif missing_pattern == 'fixed_pred':
            print(0)

        if missing_pattern != 'fixed':
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
            with open(path, "rb") as f:
                self.training_mean, self.training_std = pickle.load(f)

        # normalization using mean and std of training data
        self.data_arr_norm = ((data_arr - self.training_mean) / self.training_std) * self.actual_mask

    def create_fixed_mask(self,dim_k):
        # Define the start and end times of each range
        ranges_5min = [('05:00:00', '17:55:00')]

        # Define the date for which to generate the DatetimeIndex values
        target_date = '2023-07-01'

        # Create an empty list to store the DatetimeIndex values
        index_values = []

        # Loop through each range and generate the DatetimeIndex values within that range for the target date
        for start_time, end_time in ranges_5min:
            index_values += pd.date_range(start=f'{target_date} {start_time}', end=f'{target_date} {end_time}', freq='5min', inclusive='both').tolist()

        # Create the DatetimeIndex from the list of values
        ranges_5min = pd.DatetimeIndex(index_values)
        empty_df = pd.DataFrame(index=ranges_5min, columns=[str(i) for i in range(0, dim_k)])

        ranges_5min_obs = [('05:00:00', '07:55:00'), ('09:30:00', '12:25:00'), ('14:00:00', '16:55:00')]

        # Define the date for which to generate the DatetimeIndex values
        target_date = '2023-07-01'

        # Create an empty list to store the DatetimeIndex values
        index_values = []

        # Loop through each range and generate the DatetimeIndex values within that range for the target date
        for start_time, end_time in ranges_5min_obs:
            index_values += pd.date_range(start=f'{target_date} {start_time}', end=f'{target_date} {end_time}', freq='5min', inclusive='both').tolist()

        ranges_5min_obs = pd.DatetimeIndex(index_values)

        empty_df.loc[ranges_5min_obs, :] = 1
        empty_df.fillna(0, inplace=True)

        return empty_df.values
    
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
            
    def _generate_dow_array(self, start_date, end_date):
        # Generate the date range
        date_range = pd.date_range(start=start_date, end=end_date)

        # Generate the day of week array
        day_of_week = date_range.dayofweek

        # Convert to numpy array
        day_of_week_array = day_of_week.to_numpy()

        return (day_of_week_array, date_range)

    def __len__(self):
        if self.is_train:
            return int((len(self.data_arr_norm) - self.seq_len) / 12) + 1
        else:
            return self.test_day

    def __getitem__(self, index):
        if self.is_train:
            s_begin = index * 12
            s_end = s_begin + self.seq_len
        else:
            s_begin = 5*12 + index * 288
            s_end = s_begin + self.seq_len

        subseq = self.data_arr_norm[s_begin:s_end] # subseq has shape (seq_len, K) ndarray
        if self.missing_pattern == "fixed":
            missing_mask = self.missing_mask
        else:
            missing_mask = self.missing_mask[s_begin:s_end]
        actual_mask = self.actual_mask[s_begin:s_end]
        dow_arr = self.dow_arr[s_begin:s_end] # shape (seq_len, )
        tod_arr = self.tod_arr[s_begin:s_end] # shape (seq_len, )
        if self.spatial_inp.any():
            spatial_inp = self.spatial_inp

        sample = {
            "actual_data": subseq,
            "missing_mask": missing_mask,
            "actual_mask": actual_mask,
            "timestamps": np.arange(subseq.shape[0]),
            "dow_arr": dow_arr,
            "tod_arr": tod_arr,
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
        is_train = True,
                ):
    
    dataset = Get_Dataset(
        missing_pattern, 
        missing_rate, 
        dataset_name, 
        save_folder, 
        seq_length,
        is_train
        )

    tensor_mean = torch.from_numpy(dataset.training_mean).to(device).float()
    tensor_std = torch.from_numpy(dataset.training_std).to(device).float()

    if is_train:
        shuffle_flag = True
        drop_last = True
        # dataset = Subset(dataset, np.arange(0, int(len(dataset)*0.1)))
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = 9

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )

    return data_loader, tensor_mean, tensor_std