import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from src.architectures.linear_sequential import linear_sequential


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 hidden_dims: int = [128],
                 k_lipschitz: str = None,
                 bilipschitz: bool = False,
                 batch_norm: bool = False,
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

        self.net = linear_sequential(input_dims=[self.input_dim],
                                     hidden_dims=self.hidden_dims,
                                     output_dim=2 * self.output_dim,  # We predict mean & log var for each output dim
                                     k_lipschitz=k_lipschitz,
                                     bilipschtiz=bilipschitz,
                                     batch_norm=batch_norm,
                                     p_drop=None
                                     )

    def forward(self, x):
        soft_output_pred = self.net(x.float()).view(-1, self.output_dim, 2)
        approx_mean = soft_output_pred[:, :, 0]
        approx_var = softplus(soft_output_pred[:, :, 1])
        return approx_mean, approx_var

    # def forward(self, x):
    #     return self.net(x.float())
