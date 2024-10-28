#coding:utf-8
"""Augmented Dickey-Fuller test implemented using Pytorch"""
import math

import numpy as np
import torch
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from torch import nn, optim


def ad_fuller(series, maxlag=None):
    """Get series and return the p-value and the t-stat of the coefficient"""
    if maxlag is None:
        n = int(np.ceil(12.0 * np.power(series.shape[-1] / 100.0, 1 / 4.0)))
        n = min(series.shape[-1] // 2 - 1 - 1, n)
    elif maxlag < 1:
        n = 1
    else:
        n = maxlag

    # Putting the X values on a Tensor with Double as type
    X = torch.tensor(series)
    X = X.type(torch.DoubleTensor)

    # Generating the lagged tensor to calculate the difference
    X_1 = X.narrow(0, 1, X.shape[0] - 1)

    # Re-sizing the x values to get the difference
    X = X.narrow(0, 0, X.shape[0] - 1)
    dX = X_1 - X

    # Generating the lagged difference tensors
    # and concatenating the lagged tensors into a single one
    lagged_tensors = lagmat(dX, n)[:, :-1]
    # nobs = lagged_tensors.shape[-2]
    # lagged_tensors[:, 0] = X[-nobs - 1: -1]
    # xdshort = dX[-nobs:]
    # fullRHS = torch.cat([torch.ones((nobs, 1)), lagged_tensors], dim=-1)
    # startlag = fullRHS.shape[-1] - lagged_tensors.shape[-1] + 1
    # icbest, bestlag = _autolag(
    #     xdshort, fullRHS, startlag, maxlag
    # )

    # for i in range(1, n + 1):
    #     lagged_n = dX.narrow(0, n - i, (dX.shape[0] - n))
    #     lagged_reshape = torch.reshape(lagged_n, (lagged_n.shape[0], 1))
    #     if i == 1:
    #         lagged_tensors = lagged_reshape
    #     else:
    #         lagged_tensors = torch.cat((lagged_tensors, lagged_reshape), 1)

    # Reshaping the X and the difference tensor
    # to match the dimension of the lagged ones
    X = X.narrow(0, 0, X.shape[0] - n)
    dX = dX.narrow(0, n, dX.shape[0] - n)
    dX = torch.reshape(dX, (dX.shape[0], 1))

    # Concatenating the lagged tensors to the X one
    # and adding a column full of ones for the Linear Regression
    X = torch.cat((torch.reshape(X, (X.shape[0], 1)), lagged_tensors), 1)
    ones_columns = torch.ones((X.shape[0], 1))
    X_ = torch.cat((X, torch.ones_like(ones_columns, dtype=torch.float64)), 1)

    nobs = X_.shape[0]

    # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
    coeff = torch.mm(torch.mm(torch.inverse(
            torch.mm(torch.t(X_), X_)), torch.t(X_)), dX)

    std_error = get_std_error(X_, dX, coeff)
    coeff_std_err = get_coeff_std_error(X_, std_error, coeff)[0]
    t_stat = coeff[0]/coeff_std_err

    p_value = mackinnonp(t_stat.item(), regression="c", N=1)
    critvalues = mackinnoncrit(N=1, regression="c", nobs=nobs)
    critvalues = {
                  "1%" : critvalues[0],
                  "5%" : critvalues[1],
                  "10%" : critvalues[2]
                 }

    return t_stat.item(), p_value, n, nobs, critvalues

def get_coeff_std_error(X, std_error, p):
    """Receive the regression standard error
    and calculate for the coefficient p"""
    std_coeff = []
    for i in range(len(p)):
        s = torch.inverse(torch.mm(torch.t(X), X))[i][i] * (std_error ** 2)
        s = math.sqrt(s.item())
        std_coeff.append(s)
    return std_coeff

def get_std_error(X, label, p):
    """Get the regression standard error"""
    std_error = 0
    y_new = torch.mm(X, p)
    for i in range(len(X)):
        diff = (label[i][0] - y_new[i][0]) ** 2
        std_error += diff.item()
    std_error = math.sqrt(std_error/X.shape[0])

    return std_error

def test_shape(series, maxlag=None):
    """Get series and return the p-value and the t-stat of the coefficient"""
    if maxlag is None:
        n = int(12 * ((len(series)/100) ** (1./4)))
    elif maxlag < 1:
        n = 1
    else:
        n = maxlag

    # Putting the X values on a Tensor with Double as type
    X = torch.tensor(series)
    X = X.type(torch.DoubleTensor)

    # Generating the lagged tensor to calculate the difference
    X_1 = X.narrow(0, 1, X.shape[0] - 1)

    # Re-sizing the x values to get the difference
    X = X.narrow(0, 0, X.shape[0] - 1)
    dX = X_1 - X
    expanded_dX = toeplitz_like(dX, n)
    X = torch.cat((X.narrow(0, 0, expanded_dX.shape[0]).unsqueeze(1), expanded_dX), dim=1)

    '''# Generating the lagged difference tensors
    # and concatenating the lagged tensors into a single one
    for i in range(1, n + 1):
        lagged_n = dX.narrow(0, n - i, (dX.shape[0] - n))
        lagged_reshape = torch.reshape(lagged_n, (lagged_n.shape[0], 1))
        if i == 1:
            lagged_tensors = lagged_reshape
        else:
            lagged_tensors = torch.cat((lagged_tensors, lagged_reshape), 1)

    # Reshaping the X and the difference tensor
    # to match the dimension of the lagged ones
    X = X.narrow(0, 0, X.shape[0] - n)


    X = torch.cat((torch.reshape(X, (X.shape[0], 1)), lagged_tensors), 1)'''
    dX = dX.narrow(0, n, dX.shape[0] - n).unsqueeze(0).t()
    print(dX.shape)
    ones_columns = torch.ones((X.shape[0], 1))
    X_ = torch.cat((X, torch.ones_like(ones_columns, dtype=torch.float64)), 1)

    nobs = X_.shape[0]

    # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
    coeff = torch.mm(torch.mm(torch.inverse(
            torch.mm(torch.t(X_), X_)), torch.t(X_)), dX)

    std_error = get_std_error(X_, dX, coeff)
    coeff_std_err = get_coeff_std_error(X_, std_error, coeff)[0]
    t_stat = coeff[0]/coeff_std_err

    p_value = mackinnonp(t_stat.item(), regression="c", N=1)
    critvalues = mackinnoncrit(N=1, regression="c", nobs=nobs)
    critvalues = {
                  "1%" : critvalues[0],
                  "5%" : critvalues[1],
                  "10%" : critvalues[2]
                 }

    return t_stat.item(), p_value, n, nobs, critvalues


def toeplitz(v):
    c = v.view(-1)
    vals = torch.cat((torch.flip(c, [0]), c[1:]))
    a = torch.arange(c.shape[0]).unsqueeze(0).t()
    b = torch.arange(c.shape[0] - 1, -1, step=-1).unsqueeze(0)
    indx = a + b

    return vals[indx]

def toeplitz_like(x, n):
    r = x
    stop = x.shape[0] - 1

    if n < stop:
        stop = n

    else:
        stop = 2

    r = toeplitz(r)

    return r.narrow(1, 0, stop).narrow(0, stop - 1, r.shape[0] - stop)

def lagmat(xdiff, maxlag):
    n = len(xdiff)
    X = torch.zeros((n + maxlag, maxlag + 1))
    for i in range(maxlag + 1):
        if xdiff.shape[-1] != 1:
            xdiff = xdiff.unsqueeze(1)
        X[maxlag - i: n + maxlag - i, maxlag - i: maxlag - i + 1] = xdiff
    X = X[maxlag:n]
    return X


def _autolag(xdshort, fullRHS, startlag, maxlag):
    best_ic = float('inf')
    best_lag = None

    for lag in range(startlag, maxlag + 1):
        # 构建 lagmat 矩阵
        X = lagmat(xdshort[:, None], lag)

        # 定义模型
        class OLSModel(nn.Module):
            def __init__(self):
                super(OLSModel, self).__init__()
                self.linear = nn.Linear(lag, 1)

            def forward(self, x):
                return self.linear(x)

        model = OLSModel()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # 可以根据需要选择不同的优化器和学习率

        # 训练模型
        for epoch in range(100):  # 可以根据需要调整训练轮数
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, fullRHS[:, None])  # 损失函数可以根据需要调整
            loss.backward()
            optimizer.step()

        # 计算信息准则
        with torch.no_grad():
            predictions = model(X)
            residuals = fullRHS[:, None] - predictions
            mse = torch.mean(residuals ** 2)
            ic = mse.item() * (len(fullRHS) + lag * 2)  # 这里使用 MSE 作为信息准则，可以根据需要调整

        # 更新最佳结果
        if ic < best_ic:
            best_ic = ic
            best_lag = lag

    return best_ic, best_lag