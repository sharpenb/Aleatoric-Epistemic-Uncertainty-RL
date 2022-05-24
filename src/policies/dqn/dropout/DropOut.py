import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from src.architectures.linear_sequential import linear_sequential


class MLPDropOut(nn.Module):
    """Drop Out MLP network."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 hidden_dims: int = [128],
                 k_lipschitz: str = None,
                 bilipschitz: bool = False,
                 batch_norm: bool = False,
                 p_drop: float = .25,
                 n_samples: int = 20,
                 seed: int = 0):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_dims: size of hidden layers
        """
        super().__init__()

        torch.cuda.manual_seed(seed)

        self.input_dim = obs_size
        self.output_dim = n_actions
        self.hidden_dims = hidden_dims

        self.p_drop = p_drop
        self.n_samples = n_samples

        self.net = linear_sequential(input_dims=[self.input_dim],
                                     hidden_dims=self.hidden_dims,
                                     output_dim=2 * self.output_dim,  # We predict mean & log var for each output dim
                                     k_lipschitz=k_lipschitz,
                                     bilipschtiz=bilipschitz,
                                     batch_norm=batch_norm,
                                     p_drop=self.p_drop
                                     )

    def forward(self, x, aggregate=True):
        batch_size = x.size(0)
        assert self.n_samples > 0, "Variable n_sampled_paths negative"
        perm = torch.cat([torch.arange(self.n_samples) * batch_size + i for i in range(batch_size)], dim=0)
        repeat_per_dim = [1] * x.dim()
        repeat_per_dim[0] = self.n_samples
        duplicated_input = x.repeat(*(repeat_per_dim))[perm]
        soft_output_pred = self.net(duplicated_input.float()).view(-1,
                                                                   self.n_samples,
                                                                   self.output_dim, 2)
        if aggregate:
            approx_mean = torch.mean(soft_output_pred[:, :, :, 0], -2)
            approx_var = torch.mean(softplus(soft_output_pred[:, :, :, 1]) + 1e-10, -2)
            approx_epistemic_var = torch.var(soft_output_pred[:, :, :, 0], -2)
            return approx_mean, approx_var, approx_epistemic_var
        else:
            approx_means = soft_output_pred[:, :, :, 0]
            approx_vars = softplus(soft_output_pred[:, :, :, 1]) + 1e-10
            return approx_means, approx_vars
