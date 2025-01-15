"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""


from argparse import ArgumentParser
from multiprocessing import cpu_count


def str2bool(s):
    # s is already bool
    if isinstance(s, bool):
        return s
    # s is string repr. of bool
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    # s is something else
    else:
        return s


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def args_base():
    ap = ArgumentParser()
    # meta
    ap.add_argument(
        "--device",
        type=str,
        default="check",
        choices=["check", "cpu", "cuda", "mps"],
        help="device to run the script",
    )

    # wandb
    ap.add_argument("--wb_key", type=str, default="", help='personal wandb key')
    ap.add_argument("--wb_project", type=str, default="gfdm", help='wandb project name')
    ap.add_argument("--wb_id", type=str, default="", help='run id - randomly assigned if not specified')
    ap.add_argument("--name", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="./runs", help='dir to save checkpoints')
    ap.add_argument("--data_dir", type=str, default="./data")

    # diffusion process parameters
    ap.add_argument("--dynamics", type=str, default="fvp", choices=["fve", "fvp"], help='dynamics of diffusion process')
    ap.add_argument("--num_aug", type=int, default=2, help='number of additional processes')
    ap.add_argument("--hurst", type=float, default=0.9, help='Hurst index')
    ap.add_argument("--gamma_max", type=float, default=20.0, help='maximal gamma used for MA-fBM')
    ap.add_argument("--norm", type=str2bool, default=True, help='whether to normalize the terminal variance of the diffusion process across all values of H')

    # data parameters
    ap.add_argument(
        "--data_name", type=str, default="cifar10", choices=["cifar10", "mnist"]
    )
    ap.add_argument("--image_size", type=int, default=32)  # for mnist 28
    ap.add_argument("--channels", type=int, default=3)  # for mnist 1
    ap.add_argument("--num_classes", type=int, default=10)

    # model parameters
    ap.add_argument("--model_name", type=str, default="unet", choices=['unet'])
    ap.add_argument("--conditioning", type=str2bool, default=True)
    ap.add_argument("--model_channels", type=int, default=128)  # for mnist 64
    ap.add_argument("--num_res_blocks", type=int, default=4)  # for mnist 3
    ap.add_argument(
        "--attn_resolutions", type=list_of_ints, default=[8]
    )  # for mnist [4,2]
    ap.add_argument(
        "--channel_mult", type=list_of_ints, default=[1, 2, 2, 2]
    )  # for mnist [1, 2, 4]
    ap.add_argument("--dropout", type=float, default=0.1)  # 0.0 for mnist

    ap.add_argument("--use_ema", type=str2bool, default=True)
    return ap


def args_train(ap=None):
    ap = args_base() if ap is None else ap

    # training
    ap.add_argument(
        "--train_steps", type=int, default=10, help="number of training steps"
    )
    ap.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    ap.add_argument("--num_workers", type=int, default=min(16, cpu_count()))
    ap.add_argument("--lr", type=float, default=2e-4, help="maximal used learning rate")
    ap.add_argument(
        "--use_lr_scheduler",
        type=str2bool,
        default=True,
        help="whether to use OneCycleLR",
    )
    ap.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="clip gradients global norm to that value",
    )
    ap.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="number of batch accumulations",
    )
    ap.add_argument("--enable_progress_bar", type=str2bool, default=False)

    # wandb when to log
    ap.add_argument("--log_model", type=str2bool, default=True)
    ap.add_argument("--log_model_every_n", type=int, default=100000)

    # validation during training
    ap.add_argument("--num_sanity_val_steps", type=int, default=1)
    ap.add_argument("--val_check_interval", type=int, default=10)
    ap.add_argument(
        "--sample_steps",
        type=int,
        default=1000,
        help="steps used in reverse time sampling",
    )
    ap.add_argument("--bs_sample", type=int, default=20)

    return ap


def args_gen(ap=None):

    ap = ArgumentParser() if ap is None else ap

    ap.add_argument("--dir_num", type=str, default=None)

    # wandb parameters
    ap.add_argument("--version", type=str, default="final")
    ap.add_argument("--wb_version", type=str, default="v0")

    ap.add_argument("--pth", type=str, default=None)
    ap.add_argument("--run_id", type=int, default=000000)

    # sampling parameters
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--mode", type=str, default="sde", choices=["sde", "ode"])
    ap.add_argument("--test_batch_size", type=int, default=2)

    # showing and logging
    ap.add_argument("--show_samples", type=str2bool, default=False)
    ap.add_argument("--log_samples", type=int, default=2)

    return ap


def args_all_train():
    ap = args_base()
    ap = args_train(ap)
    return ap


def args_all_gen():
    ap = args_base()
    ap = args_gen(ap)
    return ap


if __name__ == "__main__":
    print(args_base().parse_args())
    print(args_train().parse_args())
    print(args_gen().parse_args())
