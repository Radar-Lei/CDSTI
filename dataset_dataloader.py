import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import pickle
from utils import mat2ten, ten2mat
import pandas as pd

class Get_Dataset(Dataset):
    def __init__(
            self, missing_pattern='RM', 
            missing_rate=0.3, 
            training_ratio=0.8, dataset_name="", save_folder="", 
            BM_window_length=6, seq_len=36
                ):
        """
        K: number of features, number of nodes
        D: number of days
        L_d: number of time intervals in a day
        """
        if dataset_name == "Guangzhou":
            path = "./dataset/" + dataset_name + "/tensor.mat"
            data_mat = loadmat(path)['tensor'].transpose(0,2,1)  # of shape (K, L_d, D) ndarray
            dow_arr = self._generate_dow_array('2016/08/01', '2016/09/30')[0]

        elif dataset_name == "Hangzhou":
            path = "./dataset/" + dataset_name + "/tensor.mat"
            data_mat = loadmat(path)['tensor'].transpose(0,2,1)  # of shape (K, L_d, D) ndarray
            dow_arr = self._generate_dow_array('2019/01/01', '2019/01/25')[0]

        elif dataset_name == "Seattle":
            data_arr = pd.read_pickle('./dataset/Seattle/speed_matrix_2015').values # (D*L_d, K)
            dow_arr, date_range = self._generate_dow_array('2015/01/01', '2015/12/31')
            D = len(date_range)
            L_d = 288
            data_mat = np.reshape(data_arr, (D, L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)

        elif dataset_name == "Portland":
            path = "./dataset/" + dataset_name + "/volume.npy"
            data_arr = np.load(path)
            dim1, dim2 = data_arr.shape
            dim = np.array([dim1, 96, 31])
            data_mat = mat2ten(data_arr, dim, 0) # of shape (K, L_d, D) ndarray
            dow_arr = self._generate_dow_array('2021/01/01', '2021/01/31')[0]
        else:
            print(0)

        dim1, dim2, dim3 = data_mat.shape

        dow_arr = np.repeat(dow_arr, dim2) # from (D, ) to (D*L_d, )
        tod_arr = np.concatenate([np.arange(dim2)] * dim3) # creating (L_d,) and then repeat D times to (D*L_d, )

        # reshap the data_arr to (D*L_d, K)
        # to maintain the order, first transpose (K, L_d, D) to (D, L_d, K), notice the feature dim need to place 
        # D first and then L_d since 
        # we want to the order as day first and then time points in a day, 
        data_arr = np.transpose(data_mat, (2, 1, 0)).reshape((data_mat.shape[2] * data_mat.shape[1], data_mat.shape[0]))

        data_arr[np.isnan(data_arr)] = 0

        self.training_mean = np.mean(data_arr[0:round(len(data_arr)*training_ratio)], axis = 0) # shape (K, )
        self.training_std = np.std(data_arr[0:round(len(data_arr)*training_ratio)], axis = 0) # shape (K, )
        self.training_mean[self.training_mean == 0] = 1e-6
        self.training_std[self.training_std == 0] = 1e-6

        actual_mask = 1 - (data_arr == 0) # shape (D*L_d, K)
        
        if missing_pattern == 'RM':
            np.random.seed(1000)
            data_mat_missing = data_mat * np.round(np.random.rand(dim1, dim2, dim3) + 0.5 - missing_rate) # (K, L_d, D)
        elif missing_pattern == 'NM':
            np.random.seed(1000)
            data_mat_missing = data_mat * np.round(np.random.rand(dim1, dim3) + 0.5 - missing_rate)[:, None, :]
        elif missing_pattern == 'BM':

            dim_time = dim2 * dim3
            # block window length denote the number of time points of the missing block
            np.random.seed(1000)
            vec = np.random.rand(int(dim_time / BM_window_length)) # shape (L_d*D/block_window, ), (1464,)
            temp = np.array([vec] * BM_window_length) # shape (block_window, L_d*D/block_window), (6, 1464)
            vec = temp.reshape([dim2 * dim3], order = 'F') # shape (L_d*D, ), (8760,)
            data_mat_missing = mat2ten(ten2mat(data_mat, 0) * np.round(vec + 0.5 - missing_rate)[None, :], np.array([dim1, dim2, dim3]), 0)
        else:
            data_mat_missing = data_arr
        
        # save data_mat with specified missing pattern
        save_path = save_folder + '/' + 'tensor_missing.npz'
        np.savez(save_path, data_mat_missing)

        # (K, L_d, D) -> (D, L_d, K) -> (D*L_d, K)
        data_arr_missing = np.transpose(data_mat_missing, (2, 1, 0)).reshape((data_mat_missing.shape[2] * data_mat_missing.shape[1], data_mat_missing.shape[0]))

        missing_mask = 1 - (data_arr_missing == 0) # shape (D*L_d, K)

        # normalization using mean and std of training data
        data_arr_norm = ((data_arr - self.training_mean) / self.training_std) * actual_mask

        path = save_folder + "/meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([self.training_mean, self.training_std], f)

        # split into subsequences, each subsequence has shape (seq_len, K).
        # use data_arr to keep all data, data_arr_missing is only used for creating missing masks
        self.subsequences = self._split_into_subsequences(data_arr_norm, seq_len)
        self.actual_masks = self._split_into_subsequences(actual_mask, seq_len) # each mask in ob_masks has shape (seq_len, K)
        self.missing_masks = self._split_into_subsequences(missing_mask, seq_len) 
        self.dow_arrs = self._split_into_subsequences(dow_arr, seq_len)
        self.tod_arrs = self._split_into_subsequences(tod_arr, seq_len)

    def _split_into_subsequences(self, arr, l):
        LK = arr.shape # arr could 1D or 2D
        L = LK[0]
        n_subsequences = L // l
        last_subsequence_length = L % l

        subsequences = np.split(arr[:n_subsequences * l], n_subsequences)

        if last_subsequence_length > 0:
            last_subsequence = arr[-l:]
            subsequences.append(last_subsequence)

        return subsequences
    
    def _generate_dow_array(self, start_date, end_date):
        # Generate the date range
        date_range = pd.date_range(start=start_date, end=end_date)

        # Generate the day of week array
        day_of_week = date_range.dayofweek

        # Convert to numpy array
        day_of_week_array = day_of_week.to_numpy()

        return (day_of_week_array, date_range)

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, index):
        subseq = self.subsequences[index] # subseq has shape (seq_len, K) ndarray
        missing_mask = self.missing_masks[index]
        actual_mask = self.actual_masks[index]
        dow_arr = self.dow_arrs[index] # shape (seq_len, )
        tod_arr = self.tod_arrs[index] # shape (seq_len, )

        sample = {
            "actual_data": subseq,
            "missing_mask": missing_mask,
            "actual_mask": actual_mask,
            "timestamps": np.arange(subseq.shape[0]),
            "dow_arr": dow_arr,
            "tod_arr": tod_arr
            }
        
        return sample

def get_dataloader(batch_size, 
                device, missing_pattern="RM", missing_rate=0.3, training_ratio=0.9, 
                valid_ratio=0.05, dataset_name="", save_folder="", BM_window_length=6, seq_length=36):
    
    dataset = Get_Dataset(missing_pattern, missing_rate, training_ratio, dataset_name, save_folder, BM_window_length, seq_length)

    # Define the split sizes
    train_size = int(training_ratio * len(dataset))
    valid_size = int(valid_ratio * len(dataset))

    # Split the dataset into train, validation, and test
    train_dataset = Subset(dataset, range(0, train_size))
    valid_dataset = Subset(dataset, range(train_size, train_size + valid_size))
    test_dataset = Subset(dataset, range(train_size + valid_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # tensor_mean = torch.tensor(dataset.training_mean, device=device, dtype=torch.float32)
    # tensor_std = torch.tensor(dataset.training_std, device=device, dtype=torch.float32)

    tensor_mean = torch.from_numpy(dataset.training_mean).to(device).float()
    tensor_std = torch.from_numpy(dataset.training_mean).to(device).float()

    return train_loader, valid_loader, test_loader, tensor_mean, tensor_std