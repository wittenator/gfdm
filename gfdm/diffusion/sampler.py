"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

# libs
import torch
from tqdm import tqdm
from contextlib import contextmanager

# custom libs
from gfdm.utils.operation import matrix_vector_mp, sample_from_batch_multivariate_normal


def get_sampler(sampler_name):
    if "augmented_em" in sampler_name.lower():
        sampler = AugmentedEulerMaruyama
    else:
        NotImplementedError

    return sampler


@contextmanager
def inference(model, device="cpu"):
    """
    a context manager that ensures the model is running in evaluation mode without any gradients beeing stored and on
    the specified accelerator device

    Args:
        model: a score function (torch.nn.Module)
        device: accelerator for computation
    """
    with torch.no_grad():
        model = model.eval()
        model = model.to(device)
        yield model
        model = model.train()


def get_score(score_fn, z0, t0, dif, time_weight, *args, **kwargs):
    t0 = t0.double()
    z0 = z0.double()

    if dif.K == 0:
        std = torch.sqrt(dif.brown_var(t0))[:, None, None, None]
        score = -score_fn(z0, time_weight * t0[:, None], *args, **kwargs) / std
    else:
        x_t = z0[:, :, :, :, 0].clone().float()
        y_t = z0[:, :, :, :, 1:].clone()

        sigma_t, _, corr, cov_yy, alpha, var_x, var_c = dif.marginal_stats(t0)
        s_t = torch.sum(alpha * y_t, dim=-1).float()

        assert torch.all(
            (var_x - var_c) > 0
        ), f"conditional variance can not be negative"
        cond_std = torch.sqrt(var_x - var_c)

        score = -(
            (1 / cond_std) * score_fn(x_t - s_t, time_weight * t0[:, None], *args)
        )[:, :, :, :, None]
        cond_score = torch.cat([score, -alpha * score], dim=-1)

        zeros = torch.zeros_like(x_t, device=z0.device)[:, :, :, :, None]

        # use cov_yy^{-1} @ c_t = linalg.solve(cov_yy,c_t)
        nabla_y_t = torch.linalg.solve(
            cov_yy[:, None, None, None, :, :], y_t.unsqueeze(-1)
        ).squeeze(-1)
        score_y_t = -torch.cat([zeros, nabla_y_t], dim=-1)

        score = cond_score + score_y_t

    return score


class AugmentedEulerMaruyama:
    def __init__(
        self,
        score_fn,
        dif,
        T=1.0,
        channels=3,
        size=32,
        conditioning=False,
        sample_noise=None,
        time_weight=999,
    ):

        self.score_fn = score_fn
        self.dif = dif

        self.channels = channels
        self.size = size

        self.T = torch.as_tensor([T], device=dif.device)
        self.sample_noise = sample_noise

        self.conditioning = conditioning
        self.time_weight = time_weight

    def init_process(self, batch, reverse=True, noise=None, steps=1000, dtype=torch.float64):

        device = batch.device

        if reverse:
            z_T, xi = self.reverse_init(batch, noise=noise, steps=steps)
        else:
            bs, c, h, w = batch.shape
            z_T = torch.cat(
                [
                    batch[:, :, :, :, None].to(dtype),
                    torch.zeros((bs, c, h, w, self.dif.K), device=device, dtype=dtype),
                ],
                dim=-1,
            )
            if noise is None:
                xi = torch.randn(bs, steps, c, h, w, self.dif.aug_dim, device=device, dtype=dtype)
            else:
                xi = noise
        return z_T, xi

    def reverse_init(self, batch, noise=None, steps=1000, dtype=torch.float64):

        T = self.T
        device = batch.device
        bs, c, h, w = batch.shape
        noise = (
            torch.randn(bs, steps + 1, c, h, w, device=device, dtype=dtype)
            if noise is None
            else noise[:, 0, :, :, :]
        )

        # device sanity
        noise = noise.to(device)
        if self.dif.K == 0:
            std = torch.sqrt(self.dif.brown_var(T))
            z_T = std * torch.randn(bs, c, h, w, device=device, dtype=dtype)
        else:
            cov_matrix, mean, corr, _, alpha, var_x, var_c = self.dif.marginal_stats(
                torch.ones(bs, device=device, dtype=dtype) * T
            )
            squ_cov_matrix = torch.squeeze(cov_matrix).clone()
            _, aug_dim, _ = squ_cov_matrix.shape
            z_T = sample_from_batch_multivariate_normal(
                squ_cov_matrix,
                c=c,
                h=h,
                w=w,
                batch_size=bs,
                aug_dim=aug_dim,
                device=device,
            )

        return z_T, noise

    def forward_step(self, z0, t0, xi, steps=1000):

        f = self.dif.f(z0, t0)
        G = self.dif.G(t0)
        dt = self.T / steps

        return z0 + f * dt + G * torch.sqrt(dt) * xi[:, :, :, :, None]

    def reverse_step(self, z0, t0, xi, dt, *args, on_last=False, mode="sde", **kwargs):

        if self.dif.K == 0:
            f = self.dif.mu(t0)[:, None, None, None] * z0
            G = self.dif.g(t0)[:, None, None, None]
            GG = G * G
        else:
            f = self.dif.f(z0.clone(), t0)
            G = self.dif.G(t0)
            GG = G[:, :, :, :, :, None] * G[:, :, :, :, None, :]

        score = get_score(
            self.score_fn, z0, t0, self.dif, self.time_weight, *args, **kwargs
        ).float()

        if self.dif.K == 0:
            drift = f - GG * score * (0.5 if mode == "ode" else 1.0)
        else:
            drift = f - matrix_vector_mp(GG, score) * (0.5 if mode == "ode" else 1.0)

        z_mean = z0 + drift * (-dt)
        dw = torch.sqrt(dt) * (xi[:, :, :, :, None] if not self.dif.K == 0 else xi)

        return z_mean + G * dw * (0.0 if mode == "ode" else 1.0) * (
            0.0 if on_last else 1.0
        )

    def step(
        self,
        z0,
        t0,
        xi,
        dt,
        reverse=False,
        args=[],
        on_last=False,
        mode="sde",
        kwargs={},
    ):
        return (
            self.reverse_step(
                z0, t0, xi, dt, *args, on_last=on_last, mode=mode, **kwargs
            )
            if reverse
            else self.forward_step(z0, t0, xi)
        )

    def sample(
        self,
        bs=1,
        steps=1000,
        batch=None,
        reverse=True,
        noise=None,
        mode="sde",
        eps=1e-3,
        T=1.0,
        enable_progress_bar=True,
        device="cpu",
        args=[],
        kwargs={},
    ):

        self.T = torch.as_tensor([T], device=device)

        batch = (
            torch.zeros((bs, self.channels, self.size, self.size), device=device)
            if batch is None
            else batch
        )

        z0, noise = self.init_process(batch, reverse=reverse, noise=noise, steps=steps)

        if reverse:
            t = torch.linspace(T, eps, steps, device=device) * torch.ones(
                bs, steps, device=device
            )
            dt = t[0, 0] - t[0, 1]

        else:
            t = torch.linspace(eps, T, steps, device=device) * torch.ones(
                bs, steps, device=device
            )
            dt = t[0, 1] - t[0, 0]

        t0 = t[:, 0]

        for n in tqdm(
            range(steps), desc="sampling from model", disable=~enable_progress_bar
        ):
            xi = (
                torch.randn_like(z0, device=device)
                if noise is None
                else noise[:, n + 1, :, :, :]
            )

            on_last = True if n == steps - 1 else False

            z1 = self.step(
                z0,
                t0,
                xi,
                dt,
                reverse=reverse,
                on_last=on_last,
                mode=mode,
                args=args,
                kwargs=kwargs,
            )
            z0 = z1.clone()
            if not on_last:
                t0 = t[:, n + 1]

        return z0