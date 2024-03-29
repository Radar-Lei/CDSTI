import argparse
import torch
import os
import datetime
import yaml
import json

from CDSTI_main import CDSTI
from dataset_dataloader import get_dataloader
from utils import train

"""
python experiments.py --modelfolder 'Seattle_20230521_000134_missing_pattern(RM)_misssing_rate(0.7)'
python experiments.py --dataset Guangzhou --missingpattern RM --missingrate 0.3 
python experiments.py --dataset Portland --baseconfig Portland.yaml --missingpattern BM --missingrate 0.3 --BMblocklength 4 --seqlen 18
python experiments.py --dataset PeMS7_V_228 --baseconfig PeMS7_V_228_0.3.yaml --missingpattern NRSM --missingrate 0.6

or using bash: 
bash ./scripts/Seattle.sh > $(date +'%y%m%d-%H%M%S')_Seattle_log.txt 2>&1
"""

parser = argparse.ArgumentParser(description='Conditional Diffusion Model for Spatiotemporal Traffic Data Imputation')
parser.add_argument('--dataset', type=str, default='PeMS7_V_228', help='dataset name:PeMS7_V_228, PeMS7_V_1026, Hangzhou, Seattle, or Portland')
parser.add_argument('--baseconfig', type=str, default='PeMS7_V_228_0.3.yaml', help='base config file')

parser.add_argument(
    '--missingpattern', 
    type=str, 
    default='RSM', 
    help='RSM: random structural missing, NRSM: non-random structural missing'
    )

parser.add_argument(
    '--missingrate', type=float, default=0.3, 
    help='''default missing rate: 30/%; for SM, 
    the missing rate denotes the proportion of 
    nodes or locations with empty data'''
                    )
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--nsample', type=int, default=50, help='number of samples')
parser.add_argument("--modelfolder", type=str, default="")

parser.add_argument('--seqlen', type=int, default=36, help='sequence length')

args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(current_time)

path = "./config/" + args.baseconfig
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.baseconfig == "":
    config["model"]["missing_pattern"] = args.missingpattern
    config["model"]["missing_rate"] = args.missingrate
    config["model"]["device"] = args.device
    config["model"]["sequence_length"] = args.seqlen
    config["train"]["nsample"] = args.nsample

config["model"]["save_folder"] = args.modelfolder
# folder to save the model
if args.modelfolder == "":
    foldername = (
        "./save/" + args.dataset + "_" + current_time + "_missing_pattern(" + 
        config["model"]["missing_pattern"] + ")_" + "misssing_rate(" + str(config['model']['missing_rate']) + ")" + "/"
    )
    args.modelfolder = foldername

# "./save/" + "missing_pattern(" + args.missing_pattern + ")_" + "misssing_rate(" + args.missingrate + ")" + "_" + args.modelfolder

    print('model folder:', args.modelfolder)
    os.makedirs(args.modelfolder, exist_ok=True)

if args.dataset == "PeMS7_V_228":
    spatial_dim = 228
    config["diffusion"]["spatial_dim"] = 228
elif args.dataset == "PeMS7_V_1026":
    spatial_dim = 1026
    config["diffusion"]["spatial_dim"] = 1026
elif args.dataset == "Hangzhou":
    spatial_dim = 80
elif args.dataset == "Seattle":
    spatial_dim = 323
    config["diffusion"]["spatial_dim"] = 323
elif args.dataset == "Portland":
    spatial_dim = 1156
else:
    print("No such dataset")

daily_num_samples = int(config["model"]["toddim"] / config["model"]["sequence_length"]) # 288 / 18 = 16
config["train"]["daily_num_samples"] = daily_num_samples
config["train"]["test_sample_num"] = daily_num_samples * 1 # the number of test samples, daily_num_samples * #days

# json.dumps() method can be used to convert a Python dictionary into a JSON string.
# indent: indent level in json file
print(json.dumps(config, indent=4))

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

(
    train_loader,
    test_loader,
    tensor_mean, 
    tensor_std
) = get_dataloader(
    config['train']['batch_size'], 
    config['model']['device'], 
    config['model']['missing_pattern'], 
    config['model']['missing_rate'], 
    dataset_name=args.dataset, 
    save_folder=args.modelfolder,
    seq_length = config['model']['sequence_length'],
    test_sample_num=config['train']['test_sample_num']
    )

model = CDSTI(config, config['model']['device'], spatial_dim).to(config['model']['device'])
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters in current model:", num_params)

if config["model"]["save_folder"] == "":
    train(
        model,
        config["train"],
        train_loader,
        test_loader=test_loader,
        mean = tensor_mean,
        std = tensor_std,
        foldername=args.modelfolder,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
