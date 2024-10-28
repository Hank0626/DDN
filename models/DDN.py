import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DInverse, DWT1DForward
from statsmodels.tsa.stattools import adfuller

from utils.ADF import ad_fuller as adf
from utils.learnable_wavelet import DWT1D


def gaussian_weight(size, sigma=1):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def exponential_smoothing(data, alpha):
    smoothed_data = data.clone()
    for t in range(1, data.shape[-1]):
        smoothed_data[..., t] = alpha * data[..., t] + (1 - alpha) * smoothed_data[..., t - 1]
    return smoothed_data


class DDN(nn.Module):
    def __init__(self, configs):
        super(DDN, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kernel = kernel = configs.kernel_len
        self.hkernel = hkernel = configs.hkernel_len
        self.pad = nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        if hkernel is not None:
            self.hpad = nn.ReplicationPad1d(padding=(hkernel // 2, hkernel // 2 - ((hkernel + 1) % 2)))
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type

        self.seq_len_new = self.seq_len
        self.pred_len_new = self.pred_len
        self.epsilon = 1e-5
        self._build_model()

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        args.seq_len = self.configs.seq_len
        args.pred_len = self.configs.pred_len
        args.label_len = self.configs.label_len
        args.enc_in = self.configs.enc_in
        args.dec_in = self.configs.dec_in
        args.moving_avg = 3

        args.c_out = self.configs.c_out
        self.norm_func = self.norm_sliding

        wave = self.configs.wavelet
        wave_dict = {'coif6': 17, 'coif3': 8, 'sym3': 2}
        self.len, self.j = wave_dict[wave], self.configs.j
        self.dwt = DWT1D(wave=wave, J=self.j, learnable=args.learnable)
        self.dwt_ratio = nn.Parameter(
            torch.clamp(torch.full((1, self.channels, 1), 0.), min=0., max=1.)
        )
        self.mlp = Statics_MLP(
            self.configs.seq_len, args.pd_model, args.pd_ff,
            self.configs.pred_len, drop_rate=args.dr, layer=args.pe_layers
        )

    def normalize(self, x, p_value=True):
        if self.station_type == 'adaptive':
            norm_input, seq_ms, pred_ms = self.norm(
                x=x.transpose(-1, -2)
            )
            outputs = torch.cat(pred_ms, dim=1).transpose(-1, -2)
            return norm_input.transpose(-1, -2), outputs, seq_ms
        else:
            return x, None

    def de_normalize(self, input, station_pred):
        if self.station_type == 'adaptive':
            bs, l, dim = input.shape
            mean = station_pred[..., :station_pred.shape[-1] // 2]
            std = station_pred[..., station_pred.shape[-1] // 2:]
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, l, dim)
        else:
            return input

    def norm(self, x, predict=True):
        norm_x, (seq_m, seq_s) = self.norm_func(x)
        if predict is True:
            mov_m, mov_s = self.mlp(seq_m, seq_s, x)
            if self.j > 0:
                ac, dc_list = self.dwt(x)
                norm_ac, (mac, sac) = self.norm_func(ac, kernel=self.hkernel)
                norm_dc, m_list, s_list = [], [], []
                for i, dc in enumerate(dc_list):
                    dc, (mdc, sdc) = self.norm_func(dc, kernel=self.hkernel)
                    norm_dc.append(dc)
                    m_list.append(mdc)
                    s_list.append(sdc)

                pred_m, pred_s = self.mlp(
                    self.dwt([mac, m_list], 1),
                    self.dwt([sac, s_list], 1), self.dwt([ac, dc_list], 1))

                dwt_r, mov_r = self.dwt_ratio, 1 - self.dwt_ratio
                norm_x = norm_x * mov_r + self.dwt([norm_ac, norm_dc], 1) * dwt_r
                pred_m = mov_m * mov_r + pred_m * dwt_r
                pred_s = mov_s * mov_r + pred_s * dwt_r
                return norm_x, (seq_m, seq_s), (pred_m, pred_s)
            else:
                return norm_x, (seq_m, seq_s), (mov_m, mov_s)
        return norm_x, (seq_m, seq_s)

    def norm_sliding(self, x, kernel=None):
        if kernel is None:
            kernel, pad = self.kernel, self.pad
        else:
            pad = self.hpad
        x_window = x.unfold(-1, kernel, 1)  # sliding window
        m, s = x_window.mean(dim=-1), x_window.std(dim=-1)  # acquire sliding mean and sliding standard deviation
        m, s = pad(m), pad(s)  # nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        x = (x - m) / (s + self.epsilon)  # x is stationary series
        return x, (m, s)  # m, s are non-stationary factors

    def p_value(self, x, float_type=torch.float32):
        B, ch, dim = x.shape
        p_value = adf(x.reshape(-1, dim), maxlag=min(self.kernel, 24), float_type=float_type).view(B, ch, 1)
        return p_value


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation, drop_rate=0.1, bias=False):
        super(FFN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias), activation,
            nn.Linear(d_ff, d_model, bias=bias), nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Statics_MLP(nn.Module):
    def __init__(self, seq_len, d_model, d_ff,
                 pred_len, drop_rate=0.1, bias=False, layer=1):
        super(Statics_MLP, self).__init__()
        project = nn.Sequential(nn.Linear(seq_len, d_model, bias=bias), nn.Dropout(drop_rate))
        self.m_project, self.s_project = copy.deepcopy(project), copy.deepcopy(project)
        self.mean_proj, self.std_proj = copy.deepcopy(project), copy.deepcopy(project)

        self.m_concat = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate))
        self.s_concat = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate))

        ffn = nn.Sequential(*[FFN(d_model, d_ff, nn.LeakyReLU(), drop_rate, bias) for _ in range(layer)])
        self.mean_ffn, self.std_ffn = copy.deepcopy(ffn), copy.deepcopy(ffn)

        self.mean_pred = nn.Linear(d_model, pred_len, bias=bias)
        self.std_pred = nn.Linear(d_model, pred_len, bias=bias)

    def forward(self, mean, std, x=None, x2=None):
        m_all, s_all = mean.mean(dim=-1, keepdim=True), std.mean(dim=-1, keepdim=True)
        mean_r, std_r = mean - m_all, std - s_all
        mean_r, std_r = self.mean_proj(mean_r), self.std_proj(std_r)
        if x is not None:
            m_orig, s_ori = self.m_project(x - m_all), \
                self.s_project(x if x2 is None else x2 - s_all)
            mean_r, std_r = self.m_concat(torch.cat([m_orig, mean_r], dim=-1)), \
                self.s_concat(torch.cat([s_ori, std_r], dim=-1))

        mean_r, std_r = self.mean_ffn(mean_r), self.std_ffn(std_r)
        mean_r, std_r = self.mean_pred(mean_r), self.std_pred(std_r)

        mean, std = mean_r + m_all, std_r + s_all
        return mean, F.relu(std)


def normalization(x, mean=None, std=None):
    if mean is not None and std is not None:
        return (x - mean) / std
    mean = x.mean(-1, keepdim=True).detach()
    x = x - mean
    std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
    x /= std
    return x, mean, std
