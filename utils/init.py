import argparse

import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# data
CONF.SCANNET_DIR =  os.path.join(CONF.PATH.BASE, "data/scannet/scans") # TODO change this

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--max_epoch", type=int, help="number of epochs", default=64)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--bert_learning_rate", type=float, help="bert_learning rate", default=0.00002)
    parser.add_argument("--wd", type=float, help="weight decay", default=0.05)
    parser.add_argument("--sample_points", type=int, default=50000, help="Point Number [default: 50000]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--semantic_segmentation", action="store_true", help="store_true, switch to ss")
    parser.add_argument("--use_bert", action="store_true", help="store_true")
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--output_dir', type=str, default='outputs', help='log dir')
    parser.add_argument('--save_pred', type=bool, default=False, help='save prediction')
    args = parser.parse_args()

    args.lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    
    return args