"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

# libs
import os
import torch
import wandb
import pytorch_lightning as pl
from torchvision.utils import make_grid
from ema_pytorch import EMA

# custom libs
from utils.operation import sample_from_batch_multivariate_normal, scale_img_inv
from diffusion.sampler import AugmentedEulerMaruyama


class GFDMLit(pl.LightningModule):
    def __init__(
        self,
        score_fn,
        diffusion,
        conditioning=False,
        use_ema=True,
        ema_decay=0.999,
        ema_update_every=10,
        time_weight=999,
        eps=1e-5,
        lr=2e-4,
        use_lr_scheduler=True,
        total_steps=10,
        bs_sample=10,
        sample_steps=1000,
        sampler_fn=None,
        sample_noise=None,
        size=32,
        channels=3,
        log_model_every_n=100000,
        output_dir="./model",
        wandb_logger=None,
        rank=None,
    ):
        super().__init__()

        self.score_fn = score_fn
        self.diffusion = diffusion
        self.conditioning = conditioning

        self.use_ema = use_ema
        if self.use_ema:
            self.ema = EMA(self.score_fn, beta=ema_decay, update_every=ema_update_every)

        self.time_weight = time_weight

        self.eps = eps
        self.lr = lr
        self.use_lr_scheduler = use_lr_scheduler
        self.total_steps = total_steps

        self.bs_sample = bs_sample
        self.sample_steps = sample_steps

        self.channels = channels
        self.size = size
        self.sampler_fn = sampler_fn
        self.sample_noise = sample_noise


        if self.use_ema:
            self.sampler = sampler_fn(
                self.ema,
                self.diffusion,
                channels=self.channels,
                size=self.size,
                conditioning=self.conditioning,
                sample_noise=self.sample_noise,
            )
        else:
            self.sampler = sampler_fn(
                self.score_fn,
                self.diffusion,
                channels=self.channels,
                size=self.size,
                conditioning=self.conditioning,
                sample_noise=self.sample_noise,
            )

        self.log_model_every_n = log_model_every_n
        self.output_dir = output_dir

        self.num_aug = self.diffusion.K
        self.H = round(self.diffusion.H.item(), 2)

        self.wandb_logger = wandb_logger
        self.rank = rank if rank is not None else 0
        self.current_global_step = -1

    def forward(self, x, t, *args, **kwargs):
        return self.score_fn(x, self.time_weight * t, *args, **kwargs)

    def training_step(self, batch, batch_idx):

        ############## TODO
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch, *args = batch
        #####
        else:
            args = []
        #####

        t = (
            torch.rand(batch.shape[0], device=self.rank) * (self.diffusion.T - self.eps)
            + self.eps
        )

        if self.diffusion.K == 0:
            noise = torch.randn_like(batch)
            mean, std = self.diffusion.brown_moments(batch, t)
            x_t = mean + std * noise
        else:
            cov_matrix, mean, _, _, alpha, var_x, var_c = self.diffusion.marginal_stats(
                t, batch=batch
            )
            batch_size, c, h, w = batch.shape

            std = torch.sqrt(var_x - var_c)
            z_t = sample_from_batch_multivariate_normal(
                torch.squeeze(cov_matrix),
                c=c,
                h=h,
                w=w,
                batch_size=batch_size,
                aug_dim=self.num_aug + 1,
                device=self.rank,
            )

            xi = z_t[:, :, :, :, 0].clone()
            c_t = z_t[:, :, :, :, 1:].clone()
            s_t = torch.sum(alpha * c_t, dim=-1)
            noise = (xi - s_t) / std
            x_t = xi + mean[:, :, :, :, 0] - s_t

        score = self(x_t, t, *args)
        loss = torch.square(-score + noise)
        loss = torch.mean(loss)

        key = f"H_{self.H}_K_{self.diffusion.K}_loss"
        self.log(key, loss)

        if self.global_step != self.current_global_step and self.rank == 0:
            self.current_global_step = int(self.global_step)
            if (
                self.global_step + 1
            ) % self.log_model_every_n == 0 and not self.global_step == 0:
                version = (self.global_step + 1) // self.log_model_every_n - 1
                path = f"{self.output_dir}/model"
                if not os.path.isdir(path):
                    os.makedirs(path)
                model_path = f"{path}/model-v{version}.pth"
                torch.save(self.score_fn.state_dict(), model_path)
                print(f"logged model on global step {self.global_step} to {model_path}")
                if self.use_ema:
                    ema_path = f"{path}/ema_model-v{version}.pth"
                    torch.save(self.ema.state_dict(), ema_path)
                    print(
                        f"logged ema model on global step {self.global_step} to {ema_path}"
                    )
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.rank != 0:
            return None

        if self.use_ema:
            self.ema.eval()

        args = [torch.arange(self.bs_sample).to(self.rank) % 10]
        z0 = self.sampler.sample(
            bs=self.bs_sample,
            steps=self.sample_steps,
            enable_progress_bar=False,
            device=self.device,
            args=args,
        )
        if self.diffusion.K > 0:
            samples = z0[:, :, :, :, 0]
        else:
            samples = z0

        samples = scale_img_inv(samples)
        img_grid = make_grid(samples, nrow=10)
        try:
            self.logger.log_image(key="Generation/Samples", images=[img_grid])
        except:
            print("could not log, but script will continue", flush=True)

    def load_model(
        self,
        download=True,
        from_wb=False,
        pth=None,
        ema_pth=None,
        wandb_pth=None,
        project=None,
        wb_name=None,
    ):
        if from_wb:
            file_exists = os.path.isfile(pth) if pth is not None else False
            if download and not file_exists:
                pth = self.download_model(
                    wandb_pth,
                    project=project,
                    wb_name=wb_name,
                )
                pth = os.path.join(pth, "model.ckpt")
            ckpt = torch.load(pth, map_location=self.device)
            sd = ckpt["state_dict"]
            sd = {"".join(k.split("module.")): sd[k] for k in sd.keys()}
            self.load_state_dict(sd)
        else:
            sd = torch.load(pth, map_location=self.device)
            score_sd = {"".join(k.split("module.")): sd[k] for k in sd.keys()}
            self.score_fn.load_state_dict(score_sd)

            if self.use_ema:
                sd2 = torch.load(ema_pth, map_location=self.device)
                ema_sd = {"".join(k.split("module.")): sd2[k] for k in sd2.keys()}
                self.ema.load_state_dict(ema_sd)
        print(f"model loaded to {self.device}.")
        return sd

    def sample(
        self, batch_size, steps=1000, mode="sde", eps=1e-3, enable_progress_bar=True
    ):

        with torch.no_grad():

            if self.use_ema:
                self.ema.eval()
                self.score_fn.eval()
                self.sampler = AugmentedEulerMaruyama(
                    self.ema,
                    self.diffusion,
                    channels=self.channels,
                    size=self.size,
                    sample_noise=self.sample_noise,
                    conditioning=self.conditioning,
                    time_weight=self.time_weight,
                )
            else:
                self.score_fn.eval()
                self.sampler = AugmentedEulerMaruyama(
                    self.score_fn,
                    self.diffusion,
                    channels=self.channels,
                    size=self.size,
                    sample_noise=self.sample_noise,
                    conditioning=self.conditioning,
                    time_weight=self.time_weight,
                )

            args = [torch.randint(10, (batch_size,)).to(self.device) % 10]
            labels = args[0]
            samples = self.sampler.sample(
                batch_size,
                steps=steps,
                mode=mode,
                eps=eps,
                noise=None,
                device=self.device,
                enable_progress_bar=enable_progress_bar,
                args=args,
            )

        return samples, labels

    def download_model(
        self, wandb_path, project=None, wb_name=None,
    ):
        # fetching the model
        run = wandb.init(project=project, name=wb_name)
        model = run.use_artifact(wandb_path, type="model")
        pth = model.download()
        # optional verbose notification
        print(f"downloaded {wandb_path} to {pth}")
        return pth

    def test_step(self, val_batch, batch_idx):
        self.validation_step(val_batch, batch_idx)
        return

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            self.ema.update()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, self.lr, total_steps=self.total_steps
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [opt], [scheduler]
        else:
            return opt