import math

import numpy.core.numeric as NX
import numpy as np
import torch
from numpy import polyval, poly1d
from scipy.stats import norm
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit, _tau_maxs, _tau_mins, _tau_stars, _tau_smallps, \
    _tau_largeps


def ad_fuller(series, maxlag=None, float_type=torch.float32):
    """Get series and return the p-value and the t-stat of the coefficient"""
    if maxlag is None:
        n = int(np.ceil(12.0 * np.power(series.shape[-1] / 100.0, 1 / 4.0)))
        n = min(series.shape[-1] // 2 - 1 - 1, n)
    else:
        n = maxlag
    dx = torch.diff(series, n=1, dim=-1).to(float_type).to(series.device)
    x = series.narrow(-1, 0, series.shape[-1] - 1).to(float_type).to(series.device)
    # Generating the lagged difference tensors
    # and concatenating the lagged tensors into a single one
    lagged_tensors = torch.ones((series.shape[0], dx.shape[1] - n, n)).to(float_type).to(series.device)
    for i in range(1, n + 1):
        lagged_tensors[:, :, i - 1] = dx.narrow(-1, n - i, (dx.shape[-1] - n))

    # Reshaping the X and the difference tensor to match the dimension of the lagged ones
    x = x.narrow(-1, 0, x.shape[-1] - n)
    dx = dx.narrow(-1, n, dx.shape[-1] - n)
    dx = torch.reshape(dx, (dx.shape[0], dx.shape[1], 1))

    # Concatenating the lagged tensors to the X one
    # and adding a column full of ones for the Linear Regression
    x = torch.cat((torch.reshape(x, (x.shape[0], x.shape[1], 1)), lagged_tensors), -1)
    ones_columns = torch.ones((x.shape[0], x.shape[1], 1)).to(float_type).to(series.device)
    x_ = torch.cat((x, ones_columns), -1)

    x_t = x_.transpose(-1, -2)
    x_x = x_t @ x_
    x_x = torch.linalg.pinv(x_x)
    coeff = (x_x @ x_t) @ dx

    std_error = get_std_error(x_, dx, coeff)
    coeff_std_err = get_coeff_std_error(x_x, std_error)[:, 0]

    t_stat = coeff[:, 0, 0] / torch.clamp_min(coeff_std_err, 1e-10)
    p_value = mackinnonp_(t_stat, regression="c", N=1, float_type=float_type)

    return p_value.to(torch.float32)

def get_coeff_std_error(x_x, std_error):
    X_x, std_error = x_x, std_error ** 2
    # X_x, std_error = torch.pinverse(X.transpose(-1, -2) @ X), std_error ** 2
    X_s = torch.diagonal(X_x, offset=0, dim1=-2, dim2=-1) * std_error.view(-1, 1)

    X_s = torch.sqrt(torch.clamp_min(X_s, 1e-10))

    return X_s

def get_std_error(X, label, p):
    y_new = X @ p

    diff = (label[..., 0] - y_new[..., 0]) ** 2
    std_error = diff.sum(dim=-1)
    std_error = torch.sqrt(std_error / X.shape[1])

    return std_error


def mackinnonp_(teststat, regression="c", N=1, lags=None, float_type=torch.float32):
    maxstat = _tau_maxs[regression]
    minstat = _tau_mins[regression]
    starstat = _tau_stars[regression]

    list1 = np.insert(_tau_smallps[regression][N-1], 0, None)
    list2 = _tau_largeps[regression][N-1]
    list1 = torch.tensor(list1).to(float_type).to(teststat.device)
    list2 = torch.tensor(list2).to(float_type).to(teststat.device)

    index = teststat <= starstat[N - 1]
    value = torch.zeros((teststat.shape[0], 5)).to(float_type).to(teststat.device)

    value[index, :1], value[index, 1:] = teststat[index].view(-1, 1), list1.view(1, -1)  # 3 polar
    value[~index, :1], value[~index, 1:] = teststat[~index].view(-1, 1), list2.view(1, -1)  # 4 polar

    p_value = torch.zeros_like(teststat).to(float_type).to(teststat.device)
    if value[index, :].shape[0] > 0:
        p_value[index] = torch_polyval(value[index, :]).view(-1)
    if value[~index, :].shape[0] > 0:
        p_value[~index] = torch_polyval(value[~index, :]).view(-1)

    normal_dist = torch.distributions.Normal(0, 1)
    p_value =normal_dist.cdf(p_value)

    p_value[teststat > maxstat[N-1]] = 1.0
    p_value[teststat < minstat[N-1]] = 0.0

    return p_value

def torch_polyval(x):
    p, x = x[:, 1:], x[:, :1]
    if torch.isnan(p[0, 0]):
        p = p[:, 1:]
    p = torch.flip(p, dims=[-1])

    # 计算每个系数乘以 x 的幂
    y = torch.zeros_like(x, device=x.device)
    for i in range(p.shape[-1]):
            y = y * x + p[:, i].view(-1, 1)
    return y

def np_polyval(p, x):
    y = torch.zeros_like(x, device=x.device)
    for pv in p:
        y = y * x + pv
    return y

    x = x.cpu().numpy()
    p = NX.asarray(p)
    if isinstance(x, poly1d):
        y = 0
    else:
        x = NX.asanyarray(x)
        y = NX.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y