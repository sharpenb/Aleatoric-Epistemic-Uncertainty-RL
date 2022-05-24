import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from src.architectures.linear_sequential import linear_sequential


class MLPEnsemble(nn.Module):
    """Drop Out MLP network."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 hidden_dims: int = [128],
                 k_lipschitz: str = None,
                 bilipschitz: bool = False,
                 batch_norm: bool = False,
                 n_networks: int = 20,
                 seed: int = 0,):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()

        torch.cuda.manual_seed(seed)

        self.input_dim = obs_size
        self.output_dim = n_actions
        self.hidden_dims = hidden_dims

        self.n_networks = n_networks

        self.nets = nn.ModuleList([linear_sequential(input_dims=[self.input_dim],
                                                     hidden_dims=self.hidden_dims,
                                                     output_dim=2 * self.output_dim,  # We predict mean & log var for each output dim
                                                     k_lipschitz=k_lipschitz,
                                                     bilipschtiz=bilipschitz,
                                                     batch_norm=batch_norm,
                                                     p_drop=None) for n in range(self.n_networks)])

    def forward(self, x, aggregate=True):
        batch_size = x.size(0)
        assert self.n_networks > 0, "Variable n_sampled_paths negative"
        perm = torch.cat([torch.arange(self.n_networks) * batch_size + i for i in range(batch_size)], dim=0)
        for n, net in enumerate(self.nets):
            if n == 0:
                pred = net(x.float())
            else:
                pred = torch.cat((pred, net(x.float())), dim=0)
        pred = pred[perm].view(-1, self.n_networks, self.output_dim, 2)
        if aggregate:
            approx_mean = torch.mean(pred[:, :, :, 0], -2)
            approx_var = torch.mean(softplus(pred[:, :, :, 1]) + 1e-10, -2)
            approx_epistemic_var = torch.var(pred[:, :, :, 0], -2)
            return approx_mean, approx_var, approx_epistemic_var
        else:
            approx_means = pred[:, :, :, 0]
            approx_vars = softplus(pred[:, :, :, 1]) + 1e-10
            return approx_means, approx_vars
