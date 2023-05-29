import matplotlib.pyplot as plt
import numpy as np 
import torch
import pandas as pd

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

def preprocess_data(
        samples, 
        train_mean, 
        train_std, 
        all_target,
        all_evalpoint,
        all_observed, 
        unnormalization=True
        ):
    """
    samples: shape(batch_size, n_samples, seq_len, K)
    all_target: actual data, shape (batch_size, seq_len, K)
    
    """
    
    all_target_np = all_target.cpu().numpy()
    all_evalpoint_np = all_evalpoint.cpu().numpy() # target mask, 1 for targets, 0 for cond observations
    all_observed_np = all_observed.cpu().numpy() # mask
    all_given_np = all_observed_np - all_evalpoint_np

    # SM nodes, i.e., features
    SM_inds = np.where(all_evalpoint_np[0,0,:] == 1)[0]
    print(f"\nIndices of nodes with missing values: {SM_inds}")
    samples = samples[:,:,:,SM_inds]
    all_target_np = all_target_np[:,:,SM_inds]
    all_evalpoint_np = all_evalpoint_np[:,:,SM_inds]
    all_observed_np = all_observed_np[:,:,SM_inds]
    all_given_np = all_given_np[:,:,SM_inds]

    K = samples.shape[-1] #feature
    L = samples.shape[-2] #time length

    if unnormalization:

        # notice train_mean[SM_inds] should be all zeros, but here they have values
        train_mean = train_mean[SM_inds].detach().cpu().numpy()
        train_std = train_std[SM_inds].detach().cpu().numpy()
        train_std_cuda = torch.from_numpy(train_std).cuda()
        train_mean_cuda = torch.from_numpy(train_mean).cuda()
        all_target_np=(all_target_np*train_std+train_mean)
        samples=(samples*train_std_cuda+train_mean_cuda)
    return samples, SM_inds, K, L, all_target_np, all_given_np, all_evalpoint_np

def quantile(samples, all_target_np, all_given_np):    
    qlist =[0.05,0.25,0.5,0.75,0.95]
    quantiles_imp= []
    for q in qlist:
        quantiles_imp.append(get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)
    return quantiles_imp

def plot_subplots(
        nrows, 
        ncols, 
        K, 
        L, 
        dataind, 
        quantiles_imp, 
        all_target_np, 
        all_evalpoint_np, 
        all_given_np, 
        path, 
        epoch
        ):
    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(24.0, 3.5*nrows))
    fig.delaxes(axes[-1][-1])

    for k in range(K):
        df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_given_np[dataind,:,k]})
        df2 = df2[df2.y != 0]
        row = k // 4
        col = k % 4
        axes[row][col].plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='CDI')
        axes[row][col].fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
                        color='g', alpha=0.3)
        axes[row][col].plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None')
        axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
        if col == 0:
            plt.setp(axes[row, 0], ylabel='value')
        if row == -1:
            plt.setp(axes[-1, col], xlabel='time')

    plt.savefig(f"{path}{epoch}.png")
    plt.close()



if "__main__" == __name__:
    dataset = 'Guangzhou' #choose dataset
    unnormalization = True

    nsample = 100 # number of generated sample
    foldername = "Guangzhou_20230525_152915_missing_pattern(RSM)_misssing_rate(0.1)"
    path = './save/' + foldername + '/generated_outputs_nsample' + str(nsample) + '.pk' 

    samples, SM_inds, K, L, all_target_np, all_given_np, all_evalpoint_np =  process_data(path, foldername, unnormalization)

    quantiles_imp = quantile(samples, all_target_np, all_given_np)

    ###traffic speed###
    dataind = 15 #change to visualize a different sample

    num_subplots = len(SM_inds)
    ncols = 4
    nrows = (num_subplots + ncols - 1) // ncols

    plot_subplots(nrows, ncols, K, L, dataind, quantiles_imp, all_target_np, all_evalpoint_np, all_given_np)