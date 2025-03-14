"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

import random
import string
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
import os
import wandb


def get_device(name):

    if name == 'check':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = name

    return device

def get_hurst_parameter(rough=True,brownian=True,smooth=False, extrem=True,hurst_explicit=None):

    hurst = list()
    if rough:
        if extrem:
            hurst = [0.25,0.01]
        else:
            hurst = [0.25,0.1]

    if brownian:
        hurst = [0.5] + hurst

    if smooth:
        if extrem:
            hurst = [0.99,0.75] + hurst
        else:
            hurst = [0.9,0.75] + hurst

    if hurst_explicit is not None:
        hurst = [hurst_explicit]

    if len(hurst)==0:
        hurst = [0.5]

    return hurst

def str2bool(string):
    return string == 'True'

def ds2tensor(ds, d=1):
    return ds.data if torch.is_tensor(ds.data) else torch.from_numpy(ds.data)


def save_toy_data(train_ds, test_ds, gen_ds,save_path = None, d_name='circles'):

    if save_path is None:
        save_path = f'./roughgm/data/{d_name}/'
        Path(save_path).mkdir(parents=True,exist_ok=True)

    train_path = save_path + 'train.pkl'
    torch.save(train_ds, train_path)

    test_path = save_path + 'test.pkl'
    torch.save(test_ds, test_path)

    gen_path = save_path + 'gen.pkl'
    torch.save(gen_ds, gen_path)

    return train_path, test_path, gen_path

def get_wb_init(data_name,hurst,K):
    wb_name = f'{data_name}'
    wb_name = wb_name if data_name in ['roll','moon'] else wb_name
    wb_name = '_' + wb_name + f'_H{hurst}_K{K}'
    return wb_name

def get_project_name(wb_project, dynamics, data_name, norm=True, test=False):
    if any([s in data_name.lower() for s in ["cifar", "mnist"]]):
        mode = f"on_{data_name}"
    else:
        mode = "on_toy"

    if not norm:
        name = f"{wb_project}_nonorm_{mode}_{dynamics}"
    else:
        name = f"{wb_project}_{mode}_{dynamics}"
    return name if not test else name + "_test"

def plot_alpha_and_var(var_model, N=1000, device='cpu'):
    time = var_model.u
    alpha = torch.zeros(N)
    var = torch.zeros(N)
    for step,u in enumerate(time):
        alpha, var = var_model(torch.as_tensor([u], device=device), return_alpha=True)
    plt.plot(time,torch.squeeze(alpha))
    plt.show()
    plt.plot(time, torch.squeeze(var))
    plt.show()

def init_train_wandb(args):

    print('\nwandb-key:\n',args.wb_key)

    if args.wb_key == '':
        os.environ['WANDB_DISABLED'] = 'true'

    os.environ["WANDB_DIR"] = args.output_dir
    os.environ["WANDB_CACHE_DIR"] = args.output_dir
    os.environ["WANDB_CONFIG_DIR"] = args.output_dir
    os.environ["WANDB_DISABLE_GIT"] = 'true'
    os.environ["WANDB_DISABLE_CODE"] = 'true'

    wandb.login(key=args.wb_key)

    wb_id = ''.join(random.choice(string.digits) for _ in range(6)) if args.wb_id == '' else args.wb_id
    wb_name = get_wb_init(
        args.data_name, args.hurst, args.num_aug
    )
    wb_name = wb_name if args.name == '' else args.name + '_' + wb_name
    wb_name = wb_id + wb_name

    wandb_logger = WandbLogger(
        project=args.wb_project,
        name=wb_name,
        id=wb_id,
        log_model=True,
        config=args,
        save_dir=args.output_dir,
        dir=args.output_dir,
    )

    os.environ["WANDB_DIR"] = args.output_dir
    os.environ["WANDB_CACHE_DIR"] = args.output_dir
    os.environ["WANDB_CONFIG_DIR"] = args.output_dir
    os.environ["WANDB_DISABLE_GIT"] = 'true'
    os.environ["WANDB_DISABLE_CODE"] = 'true'

    return wandb_logger, wb_name

def init_generation_wandb(args):

    print('\nwandb-key:\n', args.wb_key)

    if args.wb_key == '':
        os.environ['WANDB_DISABLED'] = 'true'

    os.environ["WANDB_DIR"] = args.output_dir
    os.environ["WANDB_CACHE_DIR"] = args.output_dir
    os.environ["WANDB_CONFIG_DIR"] = args.output_dir
    os.environ["WANDB_DISABLE_GIT"] = 'true'
    os.environ["WANDB_DISABLE_CODE"] = 'true'

    wandb.login(key=args.wb_key)

    wb_id = ''.join(random.choice(string.digits) for _ in range(6)) if args.wb_id == '' else args.wb_id

    train_name = get_wb_init(
        args.data_name, args.hurst, args.num_aug
    )
    train_dir = os.path.join(args.output_dir, str(args.run_id) + train_name)

    wb_name = f'{wb_id}_{args.run_id}_generation_H{args.hurst}_K{args.num_aug}_{args.mode}{args.steps}'
    wandb_logger = WandbLogger(project=args.wb_project, name=wb_name, id=wb_id, config=args)

    return wandb_logger, wb_name, wb_id, train_dir

def init_eval_weights_biases(args,version=None):

    print('\nwandb-key:\n', args.wb_key)

    if args.wb_key == '':
        os.environ['WANDB_DISABLED'] = 'true'

    os.environ["WANDB_DIR"] = args.output_dir
    os.environ["WANDB_CACHE_DIR"] = args.output_dir
    os.environ["WANDB_CONFIG_DIR"] = args.output_dir
    os.environ["WANDB_DISABLE_GIT"] = 'true'
    os.environ["WANDB_DISABLE_CODE"] = 'true'

    wandb.login(key=args.wb_key)

    wb_id = ''.join(random.choice(string.digits) for _ in range(6)) if args.wb_id == '' else args.wb_id
    metrics = '_'.join(metric for metric in args.metrics)
    wb_name = f'{wb_id}_{args.run_id}_eval_{metrics}_H{args.hurst}_K{args.num_aug}_{args.mode}{args.steps}'
    if version is not None:
        wb_name = f'{wb_name}_{version}'
    wandb_logger = WandbLogger(project=args.wb_project, name=wb_name, id=wb_id, config=args)
    return wandb_logger, wb_name, wb_id

def gather_metrics(args):
    metrics = []
    if args.val_wsd:
        metrics.extend(['WSD'])
    if args.val_fls:
        metrics.extend(['FLS'])
    if args.val_fid:
        metrics.extend(['FID'])
    if args.val_ipir:
        metrics.extend(['IP', 'IR'])
    if args.val_nll:
        metrics.extend(['NLL'])
    if args.val_aut:
        metrics.extend(['AuthPct'])
    if args.val_ct:
        metrics.extend(['CTScore'])
    if args.val_kid:
        metrics.extend(['KID'])
    if args.val_vs:
        metrics.extend(['VS'])
    return metrics
