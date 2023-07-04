import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from visualization import plot_subplots, quantile, preprocess_data
import os

from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import pandas as pd


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order = 'F'), 0, mode)


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
    
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def train(
    model,
    config,
    train_loader,
    test_loader=None,
    mean = 0,
    std = 1,
    valid_epoch_interval=10,
    early_stopping_patience = 10,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    # decay from 0.001 to 0.0001 and 0.00001 at 75% and 90% of total epochs
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.9, patience=50, verbose=True
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        """
        miniterval means that the progress bar will be updated 
        if at least 5 seconds have passed since the last update.

        If more than 50 seconds have passed since the last update, 
        the progress bar will be forced to update immediately. 
        This prevents the progress bar from appearing 
        unresponsive if the iteration takes a long time.

        By setting refresh=False, the progress bar will not be immediately refreshed,
        meaning the updated postfix information will not be immediately displayed.
        Instead, the progress bar will only be refreshed when the internal update interval is reached.
        This can help reduce the frequency of refreshing the progress bar, 
        which may be useful in cases where refreshing too frequently could slow down the overall execution of the loop.
        """
        with tqdm(train_loader, mininterval=2.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                # set_postfix() is used to udpate the text information after the progress bar
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_training_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            # lr_scheduler.step()
            scheduler.step(avg_loss)

        if  ((epoch_no + 1) % valid_epoch_interval == 0):
            evaluate(
                model,
                test_loader,
                config,
                nsample=config['nsample'],
                mean=mean,
                std=std,
                epoch = epoch_no,
                foldername=foldername,
            )
        
        if foldername != "":
            torch.save(model.state_dict(), output_path)

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean, std):
    target = target * std + mean
    forecast = forecast * std + mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, config, nsample=100, mean=0, std=1, epoch = 1, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        mape_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                c_target[c_target == 0] = 1e-6

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (std ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * std

                # for computing APE, i.e., MAPE, we do not need the std to unnormalize the data
                ape_current = torch.abs((samples_median.values * std - c_target * std) / (c_target * std + mean )) * eval_points

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += ape_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": (mape_total / evalpoints_total) * 100,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + "_epoch" + str(epoch) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        std,
                        mean,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean, std
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        (mape_total / evalpoints_total) * 100,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("MAPE:", (mape_total / evalpoints_total) * 100)
                print("CRPS:", CRPS)

            unnormalization = True

            # foldername = "Guangzhou_20230525_152915_missing_pattern(RSM)_misssing_rate(0.1)"
            # path = './save/' + foldername + '/generated_outputs_nsample' + str(nsample) + '.pk' 

            (
                samples, 
                K, 
                L, 
                all_target_np, 
                all_given_np, 
                all_evalpoint_np
                ) =  preprocess_data(
                all_generated_samples, 
                mean, 
                std, 
                all_target, 
                all_evalpoint, 
                all_observed_point, 
                unnormalization)

            quantiles_imp = quantile(samples, all_target_np, all_given_np)

            ###traffic speed###
            dataind = config['daily_num_samples'] # here 16, since seq_len=18, we plot the first day, 16 * 18 = 288 for PeMSD7

            # by default, num_subplots = K, but we can change it to plot less subplots
            num_subplots = min(K, 60)

            ncols = 3
            nrows = (num_subplots + ncols - 1) // ncols

            figs_path = foldername + "/figs/"
            
            os.makedirs(figs_path, exist_ok=True)

            plot_subplots(
                nrows, 
                ncols, 
                num_subplots, 
                L, 
                dataind, 
                quantiles_imp, 
                all_target_np, 
                all_evalpoint_np, 
                all_given_np, 
                figs_path, 
                epoch
                )