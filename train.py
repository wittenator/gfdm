"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

# libs
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from functools import partial
import torch

# custom libs
from args import args_all_train
from utils.base import init_train_wandb
from model_zoo.utils import get_model
from diffusion.diffusion_lib import get_diffusion
from diffusion.sampler import get_sampler
from lightning import GFDMLit as GFDM
import vision_datasets
from utils import dist_util


def run_training(
    model_fn,
    train_ds,
    wandb_logger,
    hurst,
    dynamics,
    sample_steps,
    val_check_interval,
    gamma_max,
    num_aug,
    lr,
    use_lr_scheduler,
    train_steps,
    bs_sample,
    data_name,
    conditioning,
    use_ema,
    batch_size,
    log_model_every_n,
    output_dir,
    enable_progress_bar,
    num_sanity_val_steps,
    gradient_clip_val,
    norm,
    channels=3,
    image_size=32,
    rank=0,  # rank of multi-GPU training
    world_size=1,  # number of available GPUS
    port=12345,  # port for communication
    accumulate_grad_batches=1,
    **kwargs,
):
    """
    This method provides generic training for a specific hurst index, given some dataset splits.

    Args:
        model_fn: a score model
        train_ds: training dataset
        wandb_logger: lightning wandb logger
        hurst: Hurst index
        dynamics: dynamics of diffusion process
        train_steps: number of training steps
        sample_steps: sampling steps used in validation step
        val_check_interval: frequency of validation step during training
        gamma_max: maximal gamma used for MA-fBM
        num_aug: number of additional processes
        lr: learning rate
        use_lr_scheduler: whether to use OneCycleLR
        train_steps: number of training steps
        bs_sample: sampling batch size in validation step
        data_name: name of dataset used for training
        conditioning: whether to train a class-conditional score model
        use_ema: whether to use EMA for training
        batch_size: taining batch_size
        log_model_every_n: save checkpoint every log_model_every_n training steps
        output_dir: dir to save checkpoints
        enable_progress_bar: whether to use progress bar during training
        num_sanity_val_steps: whether to do a validation step before starting the training
        gradient_clip_val: value for gradient clipping during training
        norm: whether to normalize the terminal variance of the diffusion process across all values of H
    Return:
        a trained generative fractional diffusion model
    """

    if dynamics == "fve":
        ema_decay = 0.999

    elif dynamics == "fvp":
        ema_decay = 0.9999

    device = rank
    dist_util.setup(rank, world_size=world_size, port=port)

    # setting up the model to be compatible with DDP
    model_fn = dist_util.wrap_model(model_fn, rank=rank)

    """
    training loop 
    """

    # initialize the dataloaders
    train_loader = dist_util.get_dataloader(
        train_ds, world_size=world_size, rank=rank, micro_batch_size=batch_size
    )
    val_dataloader = dist_util.get_dataloader(
        [0.0], world_size=world_size, rank=rank, micro_batch_size=1
    )

    # set up model & framework
    dif = get_diffusion(
        hurst,
        dynamics=dynamics,
        gamma_max=gamma_max,
        K=num_aug,
        norm=norm,
        device=device,
    )

    callbacks = [LearningRateMonitor(logging_interval="step")]

    sampler_fn = get_sampler("augmented_em")

    gfdm = GFDM(
        model_fn,
        dif,
        lr=lr,
        use_lr_scheduler=use_lr_scheduler,
        total_steps=train_steps,
        bs_sample=bs_sample,
        sample_steps=sample_steps,
        sampler_fn=sampler_fn,
        size=image_size,
        conditioning=conditioning,
        channels=channels,
        log_model_every_n=log_model_every_n,
        output_dir=output_dir,
        rank=rank,
        use_ema=use_ema,
        ema_decay=ema_decay,
        wandb_logger=wandb_logger,
    )

    # train the model
    trainer = Trainer(
        max_steps=train_steps,
        max_epochs=-1,
        log_every_n_steps=val_check_interval,
        check_val_every_n_epoch=None,
        val_check_interval=val_check_interval,
        devices=[rank],
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger if rank == 0 else False,
        callbacks=callbacks if rank == 0 else None,
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=gradient_clip_val,
        enable_checkpointing=False,
    )

    if rank == 0:
        print(
            f"H={hurst}: Training on {data_name} over {train_steps} steps with batch size {batch_size}..."
        )
    trainer.fit(
        gfdm, train_dataloaders=train_loader, val_dataloaders=val_dataloader,
    )
    if rank == 0:
        print("Finished training, saving final checkpoint")
        path = f"{output_dir}/model"
        if not os.path.isdir(path):
            os.makedirs(path)
        model_path = f"{path}/model-final.pth"
        torch.save(gfdm.score_fn.state_dict(), model_path)
        if use_ema:
            ema_path = f"{path}/ema_model-final.pth"
            torch.save(gfdm.ema.state_dict(), ema_path)

    print(f"H={hurst}: completed")
    return gfdm


def dist_fn_wrapper(rank, *_args, f=lambda rank: rank, args=[], kwargs={}, **_kwargs):
    """mp.spawn requires a function that takes rank as its first argument"""
    return f(*args, rank=rank, **kwargs)


def get_datasets(data_name, data_dir, **kwargs):
    """
    Returns:
        first element: training split, second element: testing split
    """
    constructor = {
        "mnist": vision_datasets.MNIST,
        "fashionmnist": vision_datasets.FASHIONMNIST,
        "cifar10": vision_datasets.CIFAR10,
    }
    return [
        constructor[data_name.lower()](train=split, cache_dir=data_dir)
        for split in (True, False)
    ]


if __name__ == "__main__":

    parser = args_all_train()
    args = parser.parse_args()

    # rephrasing the training function to be compatible with mp.spawn
    kwargs = vars(args)
    data_name = kwargs["data_name"].lower()

    # channels
    kwargs["channels"] = args.channels
    # kwargs['channels_out'] = get_channels(**kwargs)

    # loading the score model architecture
    kwargs["model_fn"] = get_model(**kwargs)

    # loading the dataset splits
    kwargs["train_ds"], kwargs["test_ds"] = get_datasets(**kwargs)

    # WandB logger initialization
    kwargs["wandb_logger"] = init_train_wandb(args)

    # create the distributed set-up
    kwargs["port"] = dist_util.find_free_port()
    world_size = torch.cuda.device_count()
    kwargs["world_size"] = world_size
    dist_training_fn = partial(dist_fn_wrapper, f=run_training, kwargs=kwargs)

    # run distributed training
    dist_util.run(dist_training_fn, world_size)
