import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model_dir", type=str, default='../model', help='The model directory')
    parser.add_argument("--checkpoint_dir", type=str, default='../checkpoint', help='The checkpoint directory')
    parser.add_argument("--model_name", type=str, default='model.pt', help='The model name')
    parser.add_argument("--log_dir", type=str, default='/shared/share_mala/andrew/OpenP5/logs', help='The log directory')
    parser.add_argument("--distributed", type=int, default=1, help='use distributed data parallel or not.')
    parser.add_argument("--gpu", type=str, default='0,1,2,3', help='gpu ids, if not distributed, only use the first one.')
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12345', help='Setup MASTER_PORT for os.environ')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def ReadLineFromFile(path):
    print(f"Attempting to read file: {path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute path: {os.path.abspath(path)}")
    
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteDictToFile(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')

                        
def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict


# def setup_logging(args):
#     args.log_name = log_name(args)
#     if len(args.datasets.split(',')) > 1:
#         folder_name = 'SP5'
#     else:
#         folder_name = args.datasets
#     folder = os.path.join(args.log_dir, folder_name)
#     if not os.path.exists(folder):
#         print(f"Attempting to create directory: {folder}")  # Add this line
#         os.makedirs(folder, exist_ok=True)  # Add exist_ok=True
#         # os.makedirs(folder)
#     log_file = os.path.join(args.log_dir, folder_name, args.log_name + '.log')
    
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)
#     logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
#     return
    

# def log_name(args):
#     if len(args.datasets.split(',')) > 1:
#         folder_name = 'SP5'
#     else:
#         folder_name = args.datasets
#     params = [str(args.distributed), str(args.sample_prompt), str(args.his_prefix), str(args.skip_empty_his), str(args.max_his), str(args.master_port), folder_name, args.tasks, args.backbone, args.item_indexing, str(args.lr), str(args.epochs), str(args.batch_size), args.sample_num, args.prompt_file[3:-4]]
#     return '_'.join(params)

def log_name(args):
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    
    # Remove path components from prompt_file, keep only filename without extension
    prompt_basename = os.path.basename(args.prompt_file)[:-4]  # removes '.txt'
    
    params = [
        str(args.distributed), 
        str(args.sample_prompt), 
        str(args.his_prefix), 
        str(args.skip_empty_his), 
        str(args.max_his), 
        str(args.master_port), 
        folder_name, 
        args.tasks, 
        args.backbone, 
        args.item_indexing, 
        str(args.lr), 
        str(args.epochs), 
        str(args.batch_size), 
        args.sample_num, 
        prompt_basename
    ]
    return '_'.join(params)


def setup_logging(args):
    args.log_name = log_name(args)
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
        
    # Create full path for log directory
    folder = os.path.join(args.log_dir, folder_name)
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder):
        print(f"Attempting to create directory: {folder}")
        os.makedirs(folder, exist_ok=True)
    
    log_file = os.path.join(folder, args.log_name + '.log')
    
    # Setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file, 
        level=args.logging_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return

def setup_model_path(args):
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    if args.model_name == 'model.pt':
        model_path = os.path.join(args.model_dir, folder_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        args.model_path = os.path.join(model_path, args.log_name+'.pt')
    else:
        args.model_path = os.path.join(args.checkpoint_dir, args.model_name)
    return
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    return
    
def load_model(model, path, args, loc=None):
    if loc is None and hasattr(args, 'gpu'):
        gpuid = args.gpu.split(',')
        loc = f'cuda:{gpuid[0]}'
    state_dict = torch.load(path, map_location=loc)
    model.load_state_dict(state_dict, strict=False)
    return model
