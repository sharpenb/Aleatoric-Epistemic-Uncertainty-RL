import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm


class BiLipschitzLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        assert input_dim == output_dim
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_linear(x)
        return y + x
