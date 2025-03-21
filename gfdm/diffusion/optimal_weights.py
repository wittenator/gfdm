"""
author: Rembert Daems
"""


import torch
import torch.special as ts


def omega_optimized(
    gamma, hurst, time_horizon, return_cost=False, return_Ab=False, device="cpu"
):
    """
    Based on mean of approximation error with type II fractional brownian motion.

    Args:
        gamma: quadrature values of the speed of mean reversion
        hurst: hurst-index
        time_horizon: time-horizon / largest time-step that is considered
        return_cost: iff specified, the cost of the approximation is returned as well
        return_Ab: if specified, the matrix A and the vector b is returned as well
    Return:
        Approximation of the quadrature values, needed to approximate fractional brownian motion in markovian setting.
    """

    gamma = torch.as_tensor(gamma, device=device)
    time_horizon = torch.as_tensor(time_horizon, device=device)

    gamma_i, gamma_j = gamma[None, :], gamma[:, None]

    A = (
        time_horizon
        + (torch.exp(-(gamma_i + gamma_j) * time_horizon) - 1) / (gamma_i + gamma_j)
    ) / (gamma_i + gamma_j)
    b = time_horizon / gamma ** (hurst + 0.5) * ts.gammainc(
        hurst + 0.5, gamma * time_horizon
    ) - (hurst + 0.5) / gamma ** (hurst + 1.5) * ts.gammainc(
        hurst + 1.5, gamma * time_horizon
    )

    # solve the linear programm
    omega = torch.linalg.solve(A, b)
    output = omega if not return_Ab else (omega, A, b)

    # return the cost if needed
    if return_cost:
        c = (
            time_horizon ** (2 * hurst + 1)
            / (2 * hurst)
            / (2 * hurst + 1)
            / torch.exp(torch.lgamma(hurst + 0.5)) ** 2
        )
        cost = 1 - b @ omega / c
        return output, cost
    else:
        return output


def gamma_by_r(num_k, r, offset=0.0, device="cpu", dtype=torch.float64):
    n = (num_k + 1) / 2 + offset
    k = torch.arange(1, num_k + 1, device=device, dtype=dtype)
    gamma = r ** (k - n)
    return gamma


def gamma_by_gamma_max(num_k, gamma_max, offset=0.0, device="cpu", dtype=torch.float64):
    r = gamma_max ** (2 / (num_k - 1 - 2 * offset))
    return gamma_by_r(num_k, r, offset, device=device, dtype=dtype)


def gamma_by_range(num_k, gamma_min, gamma_max):
    return torch.exp(torch.linspace(torch.log(gamma_min), torch.log(gamma_max), num_k))