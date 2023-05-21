import argparse
import torch
import os
import datetime
import yaml
import json

from main_model import CDSTI
from dataset_dataloader import get_dataloader
from utils import train, evaluate

"""
python experiments.py --modelfolder 'Seattle_20230521_000134_missing_pattern(RM)_misssing_rate(0.7)'
python experiments.py --dataset Guangzhou --missingpattern RM --missingrate 0.3 
python experiments.py --dataset Portland --baseconfig Portland.yaml --missingpattern BM --missingrate 0.3 --BMblocklength 4 --seqlen 18
"""

parser = argparse.ArgumentParser(description='Conditional Diffusion Model for Spatiotemporal Traffic Data Imputation')
parser.add_argument('--dataset', type=str, default='Guangzhou', help='dataset name:Guangzhou, Hangzhou, Seattle, or Portland')
parser.add_argument('--baseconfig', type=str, default='Guangzhou.yaml', help='base config file')

parser.add_argument('--missingpattern', type=str, default='BM', help='RM: random missing, NM: non-random missing, BM: blackout missing, SM: structural missing')
parser.add_argument('--missingrate', type=float, default=0.3, help='default missing rate: 30/%; for SM, the missing rate denotes the proportion of nodes or locations with empty data')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument('--BMblocklength', type=int, default=4, help='block length for blackout missing pattern')
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
    config["model"]["BM_block_window_length"] = args.BMblocklength
    config["model"]["sequence_length"] = args.seqlen

# json.dumps() method can be used to convert a Python dictionary into a JSON string.
# indent: indent level in json file
print(json.dumps(config, indent=4))

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

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

(
    train_loader, 
    valid_loader, 
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
    BM_window_length=config['model']['BM_block_window_length'],
    seq_length = config['model']['sequence_length']
    )

if args.dataset == "Guangzhou":
    spatial_dim = 214
elif args.dataset == "Hangzhou":
    spatial_dim = 80
elif args.dataset == "Seattle":
    spatial_dim = 323
elif args.dataset == "Portland":
    spatial_dim = 1156
else:
    print("No such dataset")

model = CDSTI(config, config['model']['device'], spatial_dim).to(config['model']['device'])
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters in current model:", num_params)

if config["model"]["save_folder"] == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=args.modelfolder,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=tensor_std,
    mean_scaler=tensor_mean,
    foldername=args.modelfolder,
)