# code from https://github.com/ts-kim/RevIN, with minor modifications
import pdb
import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            print('in subtract last')
            pdb.set_trace()
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            print('in subtract last')
            pdb.set_trace()
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            print('in self affine')
            pdb.set_trace()
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x


class GlobalNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, stdev: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        if mean.ndim != 1 or stdev.ndim != 1:
            raise ValueError("mean/stdev must be 1D tensors (features,)")
        self.eps = eps
        self.register_buffer("mean", mean.view(1, 1, -1))
        self.register_buffer("stdev", stdev.view(1, 1, -1))

    def forward(self, x, mode: str):
        mean = self.mean.to(x.device)
        stdev = self.stdev.to(x.device)
        if mode == "norm":
            return (x - mean) / (stdev + self.eps)
        if mode == "denorm":
            return x * (stdev + self.eps) + mean
        raise NotImplementedError

    def _denormalize(self, x):
        if self.affine:
            print('in self affine')
            pdb.set_trace()
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            print('in subtract last')
            pdb.set_trace()
            x = x + self.last
        else:
            x = x + self.mean
        return x
