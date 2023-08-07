import argparse
import torch
import random
import numpy as np
from exp_imputation import Exp_Imputation
from exp_prediction import Exp_Prediction
import datetime

"""
beta_start and beta_end and diff_steps are very important for the performance of generation
if you found the values from the decoder is exploding, you may need to reduce beta_start and beta_end, or reduce diff_steps, 
to eventually reduce the variance of the sampling distribution (diffusion rate)
"""

# random seed will control all random operations in all .py it calls
fix_seed = 20
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, default='imputation',
                    help='task name, options:[prediction, imputation]')
parser.add_argument('--is_training', type=int, default=0, help='status, options:[0:training, 1:testing, 2:pred]')
parser.add_argument('--model', type=str, default='DiffusionBase',
                        help='model name, options: [DiffusionBase, TimesNet]')

# data loader
parser.add_argument('--root_path', type=str, default='./dataset/PeMS7_228', help='root path of the data file') # ./dataset/competition/train-5min ./dataset/PeMS7_228
parser.add_argument('--pred_root_path', type=str, default='./dataset/competition/test-5min', help='root path of the test data file') # competition
parser.add_argument('--flow_data_path', type=str, default='flow-5min.csv', help='data file') # flow.csv
parser.add_argument('--speed_data_path', type=str, default='speed-5min.csv', help='data file') # speed.csv
parser.add_argument('--dataloader_type', type=str, default='flow', help='options:[agg:aggregation, flow_agg, speed_agg, flow, speed, multi: multidata but not agg]') # speed.csv
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--data_shrink', type=int, default=1, help='reduce the numbder of samples')

# imputation task
parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--missing_pattern', type=str, default='sm', 
                    help='missing pattern, options:[rm:randomly, sm:random structurally missing, sbm:structurally block missing]')
parser.add_argument('--missing_rate', type=float, default=0.3, help='missing rate')
parser.add_argument('--fixed_seed', type=int, default=20)

# model define
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

parser.add_argument('--enc_in', type=int, default=228, help='encoder input size') # dim of feature/ num of nodes
parser.add_argument('--dec_in', type=int, default=228, help='decoder input size')
parser.add_argument('--c_out', type=int, default=228, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model') # should be half of d_ff
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn') # FC network, 
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock') # 5
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception') # 6
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

# diffusion
parser.add_argument('--diff_schedule', type=str, default='quad', help='schedule for diffusion, options:[quad, linear]')
parser.add_argument('--diff_steps', type=int, default=300, help='num of diffusion steps')
parser.add_argument('--diff_samples', type=int, default=16, help='num of diffusion samples')
parser.add_argument('--beta_start', type=float, default=0.0001, help='start beta for diffusion, 0.0001')
parser.add_argument('--beta_end', type=float, default=0.01, help='end beta for diffusion, 0.1, 0.2, 0.3, 0.4')
parser.add_argument('--sampling_shrink_interval', type=int, default=4, help='shrink interval for sampling')


# optimization
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--itr', type=int, default=1, help='experiments times') # num of experiments
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multi gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.task_name == 'long_term_forecast':
    Exp = Exp_Prediction
elif args.task_name == 'imputation':
    Exp = Exp_Imputation

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.is_training == 0:
    # train
    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_dm{}_df{}_el{}_topk{}_nk{}_fq_{}_{}'.format(
            current_time,
            args.model,
            args.task_name,
            args.dataloader_type,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.top_k,
            args.num_kernels,
            args.freq,
            args.des,
            ii)
        
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting)
        torch.cuda.empty_cache() 

elif args.is_training == 2:
    # pred
    ii = 0
    setting = '20230713_033341_long_term_forecast_flow_dm256_df256_el2_topk5_nk6_fq_h_Exp_3'

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.pred(setting)
    torch.cuda.empty_cache()