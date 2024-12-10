"""

author: Maximilian Springenberg
association: Fraunhofer HHI
"""
import socket
from functools import partial
import os
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from multiprocessing import cpu_count


def find_free_port():
    """
    Returns:
        a free port number, usefull for DDP training setup
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket_name = s.getsockname()[1]
        print(f'found free socket at {socket_name}')
        return socket_name
    finally:
        s.close()


def setup(rank, world_size, backend='gloo', port=12345):#backend='gloo'):  # for CPU use backend='gloo'
    """
    sets up the distributed backend

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print('\033[94m' + f'[DIST BACKEND] setting backend of rank {rank} to {backend}' + '\033[0m', flush=True)



def wrap_model(model, rank):
    """
    Args:
        model: a torch.nn.Module
        rank: the rank of the DDP training process for that model
    Returns:
        a copy of that model on the device dedicated to the process rank
    """
    torch.cuda.empty_cache()
    #model = DDP(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model = DDP(model.to(rank), device_ids=[rank], output_device=rank)
    return model


def get_dataloader(dset, world_size, rank, seed=42, micro_batch_size=1,  **kwargs):
    """
    Args:
        dset: a torch.utils.data.Dataset
        world_size: Total number of processes
        rank: Unique identifier of each process
        seed: random seed for reproducibility
              (choose a large number with balanced bits, such as 42, it's the answer to the universe, and everything)
        micro_batch_size: the batch size of the dataloader (not necessarily the global batch size for optimization)
        **kwargs: placeholder for duck-typing
    Returns:
        A distributed dataloader, that distributes samples to the respective process rank
    """
    sampler = DistributedSampler(dset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
    return DataLoader(dset, batch_size=micro_batch_size, sampler=sampler)
    #return DataLoader(dset, batch_size=micro_batch_size, sampler=sampler)

def run(f, world_size, args=[], kwargs={}):
    """
    This method executes the main process for distributed training.

    Args:
        f: the main function (e.g. a training loop) that will be distributed to the available devices
        world_size: number of available devices
        args: list of arguments for the main function f
        kwargs: list of keyword arguments for the main function f
    """
    foo = partial(f, **kwargs)
    mp.spawn(foo, args=args, nprocs=world_size)
