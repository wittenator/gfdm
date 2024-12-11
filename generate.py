"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

# libs
import random
import string
from pathlib import Path
import einops
import os
import numpy as np
import torch
from torchvision.utils import make_grid

# custom libs
from utils.base import get_device, init_generation_wandb
from diffusion.diffusion_lib import get_diffusion
from diffusion.sampler import get_sampler
from lightning import GFDMLit as GFDM
from model_zoo.utils import get_model
from utils.operation import scale_img_inv
from args import args_all_gen

def get_save_dir(train_dir, version, mode, steps):
    return os.path.join(
        train_dir,
        'generated_data',
        f'model-{version}',
        f'{mode}{steps}'
    )

def get_model_path(train_dir, version):
    return os.path.join(train_dir, "model", f"model-{version}.pth")

def get_ema_path(train_dir, version):
    return os.path.join(
        train_dir, "model", f"ema_model-{version}.pth"
    )

def generate_data(args):

    # wandb login
    wandb_logger, wb_name, wb_id, train_dir = init_generation_wandb(args)

    # get device of current rank
    device = get_device(args.device)

    # select the right architecture config
    model_fn = get_model(**vars(args))
    # select diffusion process / dynamic
    dif = get_diffusion(
        args.hurst,
        dynamics=args.dynamics,
        gamma_max=args.gamma_max,
        K=args.num_aug,
        norm=args.norm,
        device=device,
    )
    sampler_fn = get_sampler("augmented_em")

    # model definition
    gfdm = GFDM(
        model_fn,
        dif,
        sampler_fn=sampler_fn,
        channels=args.channels,
        size=args.image_size,
        conditioning=args.conditioning,
        use_ema=args.use_ema,
    )
    gfdm.to(device)
    # retrieve model paths
    pth = get_model_path(train_dir, args.version)
    ema_pth = get_ema_path(train_dir, args.version)
    # load trained model
    gfdm.load_model(
        pth=pth,
        ema_pth=ema_pth,
        download=False,
        from_wb=False,
        project=args.wb_project,
        wb_name=wb_name,
    )
    # setting up the sampling
    n_iter = args.n_samples // args.batch_size + int(
        (args.n_samples % args.batch_size) > 0
    )
    with torch.no_grad():
        data_tensor = torch.zeros(
            n_iter,
            args.batch_size,
            args.channels,
            args.image_size,
            args.image_size,
            device=device,
        )
        label_tensor = torch.zeros(n_iter, args.batch_size, device=device)
        # batch-wise sampling
        for batch_iter in range(n_iter):
            z0, labels = gfdm.sample(args.batch_size, steps=args.steps, mode=args.mode)

            if args.num_aug > 0:
                samples = z0[:, :, :, :, 0]
            else:
                samples = z0

            samples = scale_img_inv(samples)
            img_grid = make_grid(samples, nrow=10)
            try:
                wandb_logger.log_image(key="Generation/Samples", images=[img_grid])
            except:
                print("could not log, but script will continue", flush=True)

            data_tensor[batch_iter] = samples
            label_tensor[batch_iter] = labels
        data_tensor = einops.rearrange(
            data_tensor,
            "I B C H W->(I B) C H W",
            I=n_iter,
            B=args.batch_size,
            C=args.channels,
            H=args.image_size,
            W=args.image_size,
        )
        data_tensor = data_tensor.cpu().numpy().transpose(0, 2, 3, 1)
        label_tensor = einops.rearrange(
            label_tensor, "I B -> (I B)", I=n_iter, B=args.batch_size
        )
        label_tensor = label_tensor.cpu().numpy()
        label_tensor = label_tensor[: args.n_samples]
    # saving samples
    save_dir = get_save_dir(train_dir, args.version, args.mode, args.steps)
    stamp = "".join(random.choice(string.digits) for _ in range(6))
    # path sanity
    Path(os.path.join(save_dir, "data")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(save_dir, "label")).mkdir(parents=True, exist_ok=True)
    # save all generated data
    save_data_path = os.path.join(save_dir, "data", f"data_{stamp}.npy")
    np.save(save_data_path, data_tensor)
    # save all respective classes
    save_label_path = f"{save_dir}/label/label_{stamp}.npy"
    np.save(save_label_path, label_tensor)


if __name__ == "__main__":

    parser = args_all_gen()
    args = parser.parse_args()

    generate_data(args)
