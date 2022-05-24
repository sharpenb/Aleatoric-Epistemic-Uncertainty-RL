import numpy as np
import torch.nn as nn
from src.architectures.SpectralLinear import SpectralLinear
from src.architectures.BiLipschitzLinear import BiLipschitzLinear


def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, bilipschtiz=False, p_drop=None, batch_norm=False):
    if bilipschtiz:
        assert k_lipschitz > .0
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if bilipschtiz :
            l = BiLipschitzLinear(dims[i], dims[i + 1], k_lipschitz ** (1. / num_layers))
            layers.append(l)
        elif k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    if batch_norm:
        layers.append(nn.BatchNorm1d(num_features=dims[-1]))
    return nn.Sequential(*layers)
