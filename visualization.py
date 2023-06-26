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

    # SM nodes, i.e., features (num_samples, seq_len, K)
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
        train_std_cuda = torch.from_numpy(train_std).cuda(samples.device)
        train_mean_cuda = torch.from_numpy(train_mean).cuda(samples.device)
        all_target_np=(all_target_np*train_std+train_mean)
        samples=(samples*train_std_cuda+train_mean_cuda)
    return samples, K, L, all_target_np, all_given_np, all_evalpoint_np

def process_data(
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

    # SM nodes, i.e., features (num_samples, seq_len, K)
    SM_inds = np.where(all_evalpoint_np[0,0,:] == 1)[0]

    samples = samples[:,:,:,SM_inds]
    all_target_np = all_target_np[:,:,SM_inds]
    all_evalpoint_np = all_evalpoint_np[:,:,SM_inds]
    all_observed_np = all_observed_np[:,:,SM_inds]
    all_given_np = all_given_np[:,:,SM_inds]

    K = samples.shape[-1] #feature
    L = samples.shape[-2] #time length

    if unnormalization:
        # notice train_mean[SM_inds] should be all zeros, but here they have values
        train_mean = train_mean[SM_inds]
        train_std = train_std[SM_inds]
        train_std_cuda = torch.from_numpy(train_std).cuda(samples.device)
        train_mean_cuda = torch.from_numpy(train_mean).cuda(samples.device)
        all_target_np=(all_target_np*train_std+train_mean)
        samples=(samples*train_std_cuda+train_mean_cuda)
    return samples, K, L, all_target_np, all_given_np, all_evalpoint_np

def quantile(samples, all_target_np, all_given_np):    
    qlist =[0.05,0.25,0.5,0.75,0.95]
    quantiles_imp= []
    for q in qlist:
        quantiles_imp.append(get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)
    return quantiles_imp

def plot_subplots(
        nrows, 
        ncols, 
        num_subplots, 
        L, 
        dataind, 
        quantiles_imp, 
        all_target_np, 
        all_evalpoint_np, 
        all_given_np, 
        path, 
        epoch
        ):
    """
    plot daily subplots by concatenating subseqs into daily seqs
    """
    plt.rcParams["font.size"] = 20
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(24.0, 3.5*nrows))
    # fig.delaxes(axes[-1][-1])

    for k in range(num_subplots):
        df = pd.DataFrame({"x":np.arange(0,L*dataind), "val":all_target_np[:dataind,:,k].reshape(-1), "y":all_evalpoint_np[:dataind,:,k].reshape(-1)})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x":np.arange(0,L*dataind), "val":all_target_np[:dataind,:,k].reshape(-1), "y":all_given_np[:dataind,:,k].reshape(-1)})
        df2 = df2[df2.y != 0]
        row = k // ncols
        col = k % ncols
        axes[row][col].plot(range(0,L*dataind), quantiles_imp[2][:dataind,:,k].reshape(-1), color = 'g',linestyle='solid',label='CDI')
        axes[row][col].fill_between(range(0,L*dataind), quantiles_imp[0][:dataind,:,k].reshape(-1),quantiles_imp[4][:dataind,:,k].reshape(-1),
                        color='g', alpha=0.3)
        axes[row][col].plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None', markersize=1)
        axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')

        # Get the minimum y-value from the data
        min_y = min(np.min(df.val), np.min(quantiles_imp[0][:dataind,:,k].reshape(-1)))
        max_y = max(np.max(df.val), np.max(quantiles_imp[4][:dataind,:,k].reshape(-1)))
        if min_y > 0:
            bottom = min_y - min_y*0.1
        else:
            bottom = min_y + min_y*0.1
            
        axes[row][col].set_ylim(bottom= bottom)  # Set the y-axis lower limit
        axes[row][col].set_ylim(top= max_y + max_y*0.1)  # Set the y-axis upper limit

        # axes[row][col].lengend()

        if col == 0:
            plt.setp(axes[row, 0], ylabel='Traffic Speed (mph))')
        if row == nrows-1:
            plt.setp(axes[-1, col], xlabel='Time')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
    plt.savefig(f"{path}epoch({epoch}).png",dpi=200)
    plt.close()


def plot_subplots_concat(
        nrows, 
        ncols, 
        subplots_ind_ls, 
        L, 
        dataind, 
        quantiles_imp, 
        all_target_np, 
        all_evalpoint_np, 
        path, 
        epoch,
        fig_count
        ):
    """
    plot daily subplots by concatenating subseqs into daily seqs
    """
    plt.rcParams["font.size"] = 20
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(24.0, 3.5*nrows))
    # fig.delaxes(axes[-1][-1])

    for i in range(len(subplots_ind_ls)):
        k = subplots_ind_ls[i]
        df = pd.DataFrame({"x":np.arange(0,L*dataind), "val":all_target_np[:dataind,:,k].reshape(-1), "y":all_evalpoint_np[:dataind,:,k].reshape(-1)})
        df = df[df.y != 0]

        row = i // ncols
        col = i % ncols
        axes[row][col].plot(range(0,L*dataind), 
                            quantiles_imp[2][:dataind,:,k].reshape(-1), 
                            color = 'g',
                            linestyle='solid', 
                            linewidth=3, 
                            label='deterministic estimation'
                            )
        
        axes[row][col].fill_between(range(0,L*dataind), 
                                    quantiles_imp[0][:dataind,:,k].reshape(-1),
                                    quantiles_imp[4][:dataind,:,k].reshape(-1),
                                    color='g', 
                                    alpha=0.3, 
                                    label='probabilistic estimation'
                                    )
        
        axes[row][col].plot(df.x,
                            df.val, 
                            color = 'b',
                            marker = 'o', 
                            linestyle='None', 
                            markersize=2.5, 
                            label='ground truth')

        # Get the minimum y-value from the data
        min_y = min(np.min(df.val), np.min(quantiles_imp[0][:dataind,:,k].reshape(-1)))
        max_y = max(np.max(df.val), np.max(quantiles_imp[4][:dataind,:,k].reshape(-1)))
        if min_y > 0:
            bottom = min_y - min_y*0.1
        else:
            bottom = min_y + min_y*0.1

        axes[row][col].set_ylim(bottom= bottom)  # Set the y-axis lower limit
        axes[row][col].set_ylim(top= max_y + max_y*0.1)  # Set the y-axis upper limit

        # axes[row][col].legend()

        if col == 0:
            plt.setp(axes[row, 0], ylabel='Traffic Speed (km/h)')
        if row == nrows-1:
            plt.setp(axes[-1, col], xlabel='Time')
            
    # Create a common legend for all subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, markerscale=3)

    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95, hspace=0.3, wspace=0.2)
    plt.savefig(f"{path}{fig_count}_epoch({epoch}).png",dpi=200)
    plt.close()

def plot_subplots_compare(
        nrows, 
        ncols, 
        subplots_ind_ls, 
        dataind_CDSTN,
        dataind_baseline,
        quantiles_imp_CDSTN,
        quantiles_imp_baseline,
        zip_data_CDSTN,
        zip_data_baseline,
        path, 
        epoch,
        fig_count
        ):
    """
    compare reverse process with baseline model
    """

    plt.rcParams["font.size"] = 20
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(24.0, 3.5*nrows))
    # unzip processed data
    (
        _,
        _, 
        L_CDSTN,
        all_target_np_CDSTN, 
        all_evalpoint_np_CDSTN
    ) = zip_data_CDSTN

    (
        _,
        _, 
        L_baseline,
        all_target_np_baseline, 
        all_evalpoint_np_baseline
    ) = zip_data_baseline    

    for i in range(len(subplots_ind_ls)):
        k = subplots_ind_ls[i]
        df = pd.DataFrame({"x":np.arange(0,L_CDSTN*dataind_CDSTN),
                        "val":all_target_np_CDSTN[:dataind_CDSTN,:,k].reshape(-1), 
                        "y":all_evalpoint_np_CDSTN[:dataind_CDSTN,:,k].reshape(-1)})
        df = df[df.y != 0]


        df2 = pd.DataFrame({"x":np.arange(0,L_baseline),
                        "val":all_target_np_baseline[dataind_baseline,:,k].reshape(-1), 
                        "y":all_evalpoint_np_baseline[dataind_baseline,:,k].reshape(-1)})
        df2 = df2[df2.y != 0]

        row = i // ncols
        col = i % ncols

        # using median, which is the third quantile, as the deterministic estimation result
        axes[row][col].plot(range(0,L_baseline), 
                            quantiles_imp_baseline[2][dataind_baseline,:,k].reshape(-1), 
                            color = 'orange',
                            linestyle='solid', 
                            linewidth=3, 
                            label='CSDI deterministic'
                            )
        
        axes[row][col].fill_between(range(0,L_baseline), 
                                    quantiles_imp_baseline[0][dataind_baseline,:,k].reshape(-1),
                                    quantiles_imp_baseline[4][dataind_baseline,:,k].reshape(-1),
                                    color='orange', 
                                    alpha=0.3, 
                                    label='CSDI probabilistic'
                                    )


        axes[row][col].plot(range(0,L_CDSTN*dataind_CDSTN), 
                            quantiles_imp_CDSTN[2][:dataind_CDSTN,:,k].reshape(-1), 
                            color = 'g',
                            linestyle='solid', 
                            linewidth=3, 
                            label='CDSTN deterministic'
                            )
        
        axes[row][col].fill_between(range(0,L_CDSTN*dataind_CDSTN), 
                                    quantiles_imp_CDSTN[0][:dataind_CDSTN,:,k].reshape(-1),
                                    quantiles_imp_CDSTN[4][:dataind_CDSTN,:,k].reshape(-1),
                                    color='g', 
                                    alpha=0.3, 
                                    label='CDSTN probabilistic'
                                    )
        
        axes[row][col].plot(df.x,
                            df.val, 
                            color = 'b',
                            marker = 'o', 
                            linestyle='None', 
                            markersize=2.5, 
                            label='ground truth')

        # Get the minimum y-value from the data
        min_y = min(
            np.min(df.val),
            np.min(df2.val),
            np.min(quantiles_imp_CDSTN[0][:dataind_CDSTN,:,k].reshape(-1)),
            np.min(quantiles_imp_baseline[0][dataind_baseline,:,k].reshape(-1))
            )
        
        max_y = max(
            np.max(df.val),
            np.max(df2.val),
            np.max(quantiles_imp_CDSTN[4][:dataind_CDSTN,:,k].reshape(-1)),
            np.max(quantiles_imp_baseline[4][dataind_baseline,:,k].reshape(-1))
            )
        
        if min_y > 0:
            bottom = min_y - min_y*0.1
        else:
            bottom = min_y + min_y*0.1

        axes[row][col].set_ylim(bottom= bottom)  # Set the y-axis lower limit
        axes[row][col].set_ylim(top= max_y + max_y*0.1)  # Set the y-axis upper limit

        if col == 0:
            plt.setp(axes[row, 0], ylabel='Traffic Speed (km/h)')
        if row == nrows-1:
            plt.setp(axes[-1, col], xlabel='Time')
            
    # Create a common legend for all subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, markerscale=3)

    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95, hspace=0.3, wspace=0.2)
    plt.savefig(f"{path}{fig_count}_epoch({epoch}).png",dpi=200)
    plt.close()