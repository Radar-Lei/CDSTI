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
            seq_len=36
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
            data_arr = pd.read_pickle('./dataset/Seattle/speed_matrix_2015') # (D*L_d, K)
            # data_arr.to_csv('./dataset/Seattle/speed_matrix_2015.csv', index=True)

            dow_arr, date_range = self._generate_dow_array('2015/01/01', '2015/12/31')
            D = len(date_range)
            L_d = 288
            data_mat = np.reshape(data_arr.values, (D, L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)

        elif dataset_name == "Portland":
            path = "./dataset/" + dataset_name + "/volume.npy"
            data_arr = np.load(path)
            dim1, dim2 = data_arr.shape
            dim = np.array([dim1, 96, 31])
            data_mat = mat2ten(data_arr, dim, 0) # of shape (K, L_d, D) ndarray
            dow_arr = self._generate_dow_array('2021/01/01', '2021/01/31')[0]
            
        elif dataset_name == "PeMS7_V_228":
            path = "./dataset/PeMS7/" + "PeMSD7_V_228.csv"
            data_arr = pd.read_csv(path, header=None).values # (D*L_d, K)
            date_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='D')
            # Select weekdays and exclude weekends
            weekdays = date_range[date_range.weekday < 5].dayofweek
            dow_arr = weekdays.to_numpy() # 44 weekdays
            L_d = 288 # interval 5 min

            data_mat = np.reshape(data_arr, (len(dow_arr), L_d, -1)).transpose(2, 1, 0) # (K, L_d, D)
        else:
            print(0)

        dim_K, _, dim3_D = data_mat.shape

        # reshap the data_arr to (D*L_d, K)
        # to maintain the order, first transpose (K, L_d, D) to (D, L_d, K), notice the feature dim need to place 
        # D first and then L_d since 
        # we want to the order as day first and then time points in a day, 
        data_arr = np.transpose(data_mat, (2, 1, 0)).reshape((data_mat.shape[2] * data_mat.shape[1], data_mat.shape[0]))

        data_arr[np.isnan(data_arr)] = 0

        actual_mask = 1 - (data_arr == 0) # shape (D*L_d, K)
        
        if missing_pattern == 'RSM':
            np.random.seed(1000)
            selected_features = np.random.choice(dim_K, round(dim_K * missing_rate), replace=False)
            data_arr_missing = data_arr.copy()
            data_arr_missing[:, selected_features] = 0 # shape (D*L_d, K)

        elif missing_pattern == 'NRSM':
            np.random.seed(1000)
            data_mat_missing = data_mat * np.round(np.random.rand(dim_K, dim3_D) + 0.5 - missing_rate)[:, None, :]
        else:
            # this missing pattern should be called mixed, but not usre if it's meaningful
            """
            I think this senario is still meaningful, in reality, if we wanna do network-wide estimation, 
            the observed data we use to infer the SM could be partially missing. we need to test the model
            performance/power under this senario.
            """
            data_mat_missing = data_arr
        
        missing_mask = 1 - (data_arr_missing == 0) # shape (D*L_d, K), 1 indicates observed, 0 indicates missing
        # save the missing mask for method comparison
        save_path = save_folder + '/' + 'tensor_missing.npz'
        np.savez(save_path, missing_mask)

        # both are of shape (K, )
        self.training_mean, self.training_std = self._meanstd_calculator(data_arr, actual_mask) 

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

    def _meanstd_calculator(self, arr, mask):
        _, dim2 = arr.shape # arr's shape (D*L_d, K)
        mean = np.zeros(dim2)
        std = np.zeros(dim2)

        for k in range (dim2):
            k_arr = arr[:, k][mask[:, k] == 1]
            mean[k] = k_arr.mean()
            std[k] = k_arr.std()
        
        return mean, std
            
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

        sample = {
            "actual_data": subseq,
            "missing_mask": missing_mask,
            "actual_mask": actual_mask,
            "timestamps": np.arange(subseq.shape[0]),
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
        test_sample_num= 1
                ):
    
    dataset = Get_Dataset(
        missing_pattern, 
        missing_rate, 
        dataset_name, 
        save_folder, 
        seq_length
        )
    
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    # np.random.seed(1000)
    # np.random.shuffle(indices)

    test_size = int(test_sample_num)
    test_indices = indices[:test_size]

    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    tensor_mean = torch.from_numpy(dataset.training_mean).to(device).float()
    tensor_std = torch.from_numpy(dataset.training_std).to(device).float()

    return train_loader, test_loader, tensor_mean, tensor_std