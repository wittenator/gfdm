"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

# libs
import torch
import numpy as np
from scipy.integrate import solve_ivp
from torch.distributions import MultivariateNormal

# custom libs
from abc import ABC, abstractmethod
import torch.nn as nn
import einops

from gfdm.diffusion.optimal_weights import omega_optimized, gamma_by_gamma_max, gamma_by_r


def device_sanity(*args, device="cuda"):
    """
    enforeces all arguments to share the same accelerator device

    Args:
        *args: a listing of tensors or modules
        device: accelerator device to share (e.g. cuda, cpu, mps, ...)
    Return:
        tensors and modules on specified device, ordering is upheld
    """
    outs = []
    for a in args:
        try:
            outs.append(a.to(device))
        except:
            pass
    return outs


def get_diffusion(
    H,
    dynamics="fve",
    gamma_max=20.0,
    K=0,
    T=1.0,
    norm=True,
    device="cpu",
    dtype=torch.float64,
):
    """
    Diffusion dynamics constructor

    Args:
        H: hurst index
        dynamics: name of the dynamic to udr (choose from ['fve','fvp'])
        gamma_max:
        K: number of additional processes
        T: terminal time
        norm: whether to normalize the terminal variance of the diffusion process across all values of H
        device: accelerator device to compute
    Return:
        the diffusion process
    """

    constructors = {
        "fve": FVE,
        "fvp": FVP,
    }

    return constructors[dynamics.lower()](
        H=H,
        gamma_max=gamma_max,
        K=K,
        T=T,
        norm=norm,
        device=device,
        dtype=dtype,
    )


class FractionalDiffusion(ABC, nn.Module):

    """Abstract class for fractional diffusion processes"""

    def __init__(self, H=0.5, gamma_max=20.0, K=5, T=1.0, pd_eps=1e-4, device="cpu", dtype=torch.float64):
        super(FractionalDiffusion, self).__init__()

        """parameters of fBM approximation"""
        self.register_buffer("H", torch.as_tensor(H, device=device))
        self.register_buffer("gamma_max", torch.as_tensor(gamma_max, device=device))
        self.register_buffer("T", torch.as_tensor([T], device=device))
        self.K = K
        self.dtype = dtype

        """parameters of augmented process"""
        self.aug_dim = K + 1
        self.pd_eps = pd_eps

        self.device = device

        if self.K > 0:
            if self.K == 1:
                gamma = gamma_by_r(
                    K, torch.sqrt(torch.tensor(gamma_max)), device=device, dtype=dtype
                )
            else:
                gamma = gamma_by_gamma_max(K, self.gamma_max, device=device, dtype=dtype)
            omega, A, b = omega_optimized(
                gamma, self.H, self.T, return_Ab=True, device=device
            )

        else:
            gamma = torch.tensor([0.0])
            omega = torch.tensor([1.0])
            A = torch.tensor([1.0])
            b = torch.tensor([1.0])

        self.register_buffer("gamma", torch.as_tensor(gamma, device=device)[None, :])
        self.register_buffer("gamma_i", self.gamma[:, :, None].clone())
        self.register_buffer("gamma_j", self.gamma[:, None, :].clone())

        self.update_omega(omega, A=A, b=b)

    def update_omega(self, omega, A=None, b=None):

        if A is not None:
            self.register_buffer("A", torch.as_tensor(A, device=self.device))
        if b is not None:
            self.register_buffer("b", torch.as_tensor(b, device=self.device))

        self.register_buffer(
            "omega", torch.as_tensor(omega, device=self.device)[None, :].clone()
        )
        self.register_buffer("sum_omega", torch.sum(self.omega))
        self.register_buffer("omega_i", self.omega[:, :, None].clone())
        self.register_buffer("omega_j", self.omega[:, None, :].clone())
        self.double_sum_omega = torch.sum(self.omega_i * self.omega_j, dim=(1, 2))

    def mean_scale(self, t):
        return torch.exp(self.integral(t))

    def mean(self, x0, t):
        c_t = self.mean_scale(t)[:, None, None, None, None]
        bs, c, h, w = x0.shape
        return torch.cat(
            [
                (c_t * x0[:, :, :, :, None]),
                torch.zeros(bs, c, h, w, self.K, device=x0.device),
            ],
            dim=-1,
        )

    def brown_moments(self, x0, t):
        return (
            self.mean_scale(t)[:, None, None, None] * x0,
            torch.sqrt(self.brown_var(t))[:, None, None, None],
        )

    def augmented_var(self, t):
        return torch.diagonal(self.cov(t), dim1=1, dim2=2)

    def forward_var(self, t):
        return self.augmented_var(t)[:, 0]

    def f(self, z0, t):
        bs = t.shape[0]
        F_t = torch.cat([self.mu(t)[:, None], -self.gamma.repeat(bs, 1)], dim=-1)[
            :, None, None, None, :
        ]
        z1 = F_t * z0
        z1[:, :, :, :, 0] = z1[:, :, :, :, 0] + self.g(t)[
            :, None, None, None
        ] * torch.sum(self.omega[:, None, None, None, :] * z1[:, :, :, :, 1:], dim=-1)
        return z1

    def G(self, t):
        M = 1 if len(t.shape) == 0 else t.shape[0]
        return torch.cat(
            [
                (self.sum_omega * self.g(t))[:, None, None, None, None],
                torch.ones(M, self.K, device=t.device)[:, None, None, None, :],
            ],
            dim=-1,
        )

    def prior_logp(self, z):
        if self.K == 0:
            shape = z.shape
            N = np.prod(shape[1:])
            var_T = self.brown_var(self.T).detach().cpu().item()
            logp = -N / 2.0 * np.log(2 * np.pi * var_T) - torch.sum(
                z ** 2, dim=(1, 2, 3)
            ) / (2.0 * var_T)
        else:
            logp = self.terminal(z)
        return logp

    def marginal_stats(self, t, batch=None):

        eps = self.pd_eps
        mean = self.mean(batch, t) if batch is not None else None
        cov = self.cov(t)
        bs = cov.shape[0]
        sigma_t = torch.squeeze(cov).clone().to(t.device)

        if bs == 1:
            sigma_t = sigma_t[None, :, :]

        I_eps = torch.eye(self.aug_dim, self.aug_dim, device=t.device)[
            None, :, :
        ] * torch.ones((t.shape[0], self.aug_dim, self.aug_dim), device=t.device)
        I_eps[:, 1:, 1:] = I_eps[:, 1:, 1:] * (
            eps * torch.exp(-2 * self.gamma * t[:, None])[:, :, None]
        )
        I_eps[:, 0, 0] = 0.0
        sigma_t = sigma_t + I_eps

        corr = sigma_t[:, 1:, 0].clone()
        cov_yy = sigma_t[:, 1:, 1:].clone()
        var_x = sigma_t[:, 0, 0].clone()
        alpha = torch.linalg.solve(cov_yy, corr)
        var_c = torch.sum(alpha * corr, dim=-1)

        return (
            sigma_t[:, None, None, None],
            mean,
            corr,
            cov_yy,
            alpha[:, None, None, None, :],
            var_x[:, None, None, None],
            var_c[:, None, None, None],
        )

    def compute_YiYj(self, t):
        sum_gamma = self.gamma_i + self.gamma_j
        return (1 - torch.exp(-t * sum_gamma)) / sum_gamma

    def numpy_compute_YiYj(self, t):
        gamma_i, gamma_j = (
            self.gamma[0, :, None].cpu().numpy(),
            self.gamma[0, None, :].cpu().numpy(),
        )
        return (1 - np.exp(-(gamma_i + gamma_j) * t.cpu().numpy())) / (
            gamma_i + gamma_j
        )

    def func(self, t, S):
        num_k = self.K
        t = torch.as_tensor(t, dtype=self.dtype)
        A = np.zeros((num_k + 1, num_k + 1), dtype=np.float64)
        A[0, 0] = 2 * self.mu(t).cpu().numpy()
        A[0, 1:] = -2 * (self.g(t) * self.omega[0] * self.gamma[0]).cpu().numpy()
        A[1:, 1:] = np.diag((self.mu(t) - self.gamma[0]).cpu().numpy())
        b = np.zeros(num_k + 1)
        b[0] = (self.omega[0].cpu().numpy().sum() * self.g(t).cpu().numpy()) ** 2
        b[1:] = self.g(t).cpu().numpy() * (
            self.omega[0].cpu().numpy().sum()
            - self.numpy_compute_YiYj(t) @ (self.omega[0] * self.gamma[0]).cpu().numpy()
        )

        return A @ S + b

    def solve_cov_ode(self):
        S_0 = np.zeros(self.K + 1)
        self.approx_sigma = solve_ivp(self.func, (0.0, 1.0), S_0, dense_output=True)

    @abstractmethod
    def mu(self, t):
        pass

    @abstractmethod
    def g(self, t):
        pass

    @abstractmethod
    def integral(self, t):
        pass

    @abstractmethod
    def brown_var(self, t):
        pass

    @abstractmethod
    def compute_cov(self, t):
        pass

    @abstractmethod
    def cov(self, t):
        pass

    @abstractmethod
    def terminal(self, z):
        pass


class FVE(FractionalDiffusion):

    """Variance exploding fractional diffusion process"""

    def __init__(
        self,
        sigma_min=0.01,
        sigma_max=50.0,
        H=0.5,
        gamma_max=20.0,
        K=5,
        T=1.0,
        norm=False,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(H=H, gamma_max=gamma_max, K=K, T=T, device=device, dtype=dtype)

        print(f"DEVICE IN FVE={device}")

        self.name = "fve"
        self.register_buffer(
            "sigma_min", torch.as_tensor(torch.tensor([sigma_min]), device=self.device)
        )
        self.register_buffer(
            "sigma_max", torch.as_tensor(torch.tensor([sigma_max]), device=self.device)
        )

        # eq. (203)
        self.register_buffer(
            "r", torch.as_tensor(self.sigma_max / self.sigma_min, device=self.device)
        )
        self.register_buffer(
            "a",
            torch.as_tensor(
                self.sigma_min
                * torch.sqrt(
                    2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min))
                ),
                device=self.device,
            ),
        )

        self.norm = norm

        if self.norm and self.K > 0:
            var_T = self.compute_covXiXj(self.T[:, None, None])
            omega = self.sigma_max * self.omega[0] / torch.sqrt(var_T)
            self.update_omega(omega)

    def mu(self, t):
        return 0 * t

    def g(self, t):
        return self.a * (self.r ** t)

    def integral(self, t):
        return 0 * t

    def brown_var(self, t):
        return (self.sigma_min ** 2) * ((self.sigma_max / self.sigma_min) ** (2 * t))

    def terminal(self, z):

        shape = z.shape
        if len(shape) == 4:
            log_prob = self.prior_logp(z)
        else:
            bs, c, h, w, aug_dim = z.shape
            sigma_T = torch.squeeze(self.marginal_stats(self.T)[0])
            mvn = MultivariateNormal(torch.zeros(aug_dim, device=z.device), sigma_T)
            z = einops.rearrange(
                z, "B C H W K->(B C H W) K ", B=bs, C=c, H=h, W=w, K=aug_dim
            )
            log_prob = mvn.log_prob(z)
            log_prob = einops.rearrange(
                log_prob, "(B C H W) -> B (C H W)", B=bs, C=c, H=h, W=w
            )
            log_prob = torch.sum(log_prob, dim=1)
        return log_prob

    def compute_cov(self, t):

        """Computes covariance matrix according to Appendix B Forward sampling"""

        bs = t.shape[0]
        sigma_t = torch.zeros(bs, self.aug_dim, self.aug_dim)
        XYk = self.compute_XYl(t, self.omega_i, self.gamma_i, self.gamma_j)
        sigma_t[:, 0, 0] = self.compute_covXX(t)
        sigma_t[:, 1:, 0] = XYk.clone()
        sigma_t[:, 0, 1:] = XYk.clone()
        sigma_t[:, 1:, 1:] = self.compute_YiYj(t)

        return sigma_t[:, None, None, None, :, :]

    def cov(self, t):
        t = t[None, None, None] if len(t.shape) == 0 else t[:, None, None]
        return self.compute_cov(t)

    def compute_Ik(self, t, gamma_k):

        """Implements eq. (106) - eq. (107)"""

        a = self.a.clone()
        r = self.r.clone()

        part1 = (((a ** 2) * gamma_k) / (torch.log(r) - gamma_k)) * (
            ((r ** (2 * t)) - ((r ** t) * torch.exp(-gamma_k * t)))
            / (torch.log(r) + gamma_k)
        )
        part2 = (((a ** 2) * gamma_k) / (torch.log(r) - gamma_k)) * (
            (r ** (2 * t) - 1) / (2 * torch.log(r))
        )
        return part1 - part2

    def compute_Iij(self, t, gamma_i, gamma_j):

        """Implements eq. (110) - eq. (111)"""

        a = self.a.clone()
        r = self.r.clone()

        scale = ((a ** 2) * (gamma_i * gamma_j)) / (
            (torch.log(r) - gamma_i) * (torch.log(r) - gamma_j)
        )
        part1 = (r ** (2 * t)) * (
            (1 - torch.exp(-(gamma_i + gamma_j) * t)) / (gamma_i + gamma_j)
        )
        part2 = ((r ** (2 * t)) - ((r ** t) * torch.exp(-gamma_i * t))) / (
            torch.log(r) + gamma_i
        )
        part3 = ((r ** (2 * t)) - ((r ** t) * torch.exp(-gamma_j * t))) / (
            torch.log(r) + gamma_j
        )
        part4 = ((r ** (2 * t)) - 1) / (2 * torch.log(r))
        return scale * (part1 - part2 - part3 + part4)

    def compute_covXiXj(self, t):

        """Calculates the variance of X_t for FVE dynamics by eq. (91) - eq. (97)"""

        offset = self.sigma_min ** 2
        Ii = self.compute_Ik(t[:, :, 0], self.gamma_i[:, :, 0])[:, :, None]
        Ij = self.compute_Ik(t[:, 0, :], self.gamma_j[:, 0, :])[:, None, :]
        Iij = self.compute_Iij(t, self.gamma_i, self.gamma_j)
        omega_ij = self.omega_i * self.omega_j
        return (
            torch.sum(
                omega_ij * (self.brown_var(t) - offset + (Iij - (Ii + Ij))), dim=(1, 2)
            )
            + offset
        )

    def compute_covXX(self, t):
        return self.compute_covXiXj(t)

    def compute_XYl(self, t, omega_k, gamma_k, gamma_l):

        """Calculates the covariance of X and Y^{l} for FVE dynamics by eq. (112) - eq. (117)"""

        a = self.a.clone()
        r = self.r.clone()

        part1 = (a / (torch.log(r) + gamma_l)) * ((r ** t) - torch.exp(-gamma_l * t))
        part2 = (
            ((a * gamma_k) / (torch.log(r) - gamma_k))
            * ((r ** t) / (gamma_k + gamma_l))
            * (1 - torch.exp(-(gamma_k + gamma_l) * t))
        )
        part3 = (
            ((a * gamma_k) / (torch.log(r) - gamma_k))
            * (1 / (torch.log(r) + gamma_l))
            * ((r ** t) - torch.exp(-gamma_l * t))
        )
        return torch.sum(omega_k * (part1 - part2 + part3), dim=1)


class FVP(FractionalDiffusion):

    """Variance preserving fractional diffusion process"""

    def __init__(
        self,
        beta_min=0.1,
        beta_max=20.0,
        H=0.5,
        gamma_max=20.0,
        K=5,
        T=1.0,
        norm=True,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(H=H, gamma_max=gamma_max, K=K, T=T, device=device, dtype=dtype)

        self.name = "fvp"
        self.register_buffer(
            "beta_min", torch.as_tensor(torch.tensor([beta_min]), device=self.device)
        )
        self.register_buffer(
            "beta_max", torch.as_tensor(torch.tensor([beta_max]), device=self.device)
        )

        self.norm = norm

        if self.K > 0:
            self.solve_cov_ode()

        if self.norm and self.K > 0:
            var_T = self.compute_cov(self.T[:, None, None])[0, 0, 0, 0, 0, 0]
            omega = self.omega[0] / torch.sqrt(var_T)
            self.update_omega(omega)
            self.solve_cov_ode()

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mu(self, t):
        return -0.5 * self.beta(t)

    def g(self, t):
        return torch.sqrt(self.beta(t))

    def integral(self, t):
        return (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )

    def brown_var(self, t):
        scale = -0.5 * (t ** 2) * (self.beta_max - self.beta_min) - t * self.beta_min
        return 1 - torch.exp(scale)

    def cov(self, t):
        t = t[None, None, None] if len(t.shape) == 0 else t[:, None, None]
        return self.compute_cov(t)

    def compute_cov(self, t):

        S = self.approx_sigma.sol(t[:, 0, 0].cpu().numpy())
        cov = np.zeros((self.K + 1, self.K + 1, t.shape[0]))
        cov[0, :, :] = S
        cov[:, 0, :] = S
        sigma_t = torch.from_numpy(cov).to(t.device).permute(2, 1, 0)
        sigma_t[:, 1:, 1:] = self.compute_YiYj(t)
        return sigma_t[:, None, None, None, :, :]

    def terminal(self, z):

        shape = z.shape
        if len(shape) == 4:
            log_prob = self.prior_logp(z)
        else:
            bs, c, h, w, aug_dim = z.shape
            sigma_T = torch.squeeze(self.marginal_stats(self.T)[0])
            mvn = MultivariateNormal(torch.zeros(aug_dim, device=z.device), sigma_T)
            z = einops.rearrange(
                z, "B C H W K->(B C H W) K ", C=c, H=h, W=w, B=bs, K=aug_dim
            )
            log_prob = mvn.log_prob(z)
            log_prob = einops.rearrange(
                log_prob, "(B C H W) -> B (C H W)", B=bs, C=c, H=h, W=w
            )
            log_prob = torch.sum(log_prob, dim=1)
        return log_prob