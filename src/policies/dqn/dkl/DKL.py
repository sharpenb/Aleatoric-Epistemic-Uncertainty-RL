import gym
import torch
from torch import Tensor, nn

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    # IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from src.policies.dqn.dkl.IndependentMultitaskVariationalStrategy import IndependentMultitaskVariationalStrategy

from sklearn import cluster

from src.architectures.linear_sequential import linear_sequential


class MLPDKL(nn.Module):
    """Drop Out MLP network."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 hidden_dims: int = [128],
                 k_lipschitz: str = None,
                 bilipschitz: bool = False,
                 batch_norm: bool = False,
                 latent_dim: int = 32,
                 n_inducing_points: int = 40,
                 kernel: str = "RBF",
                 env: gym.Env = None,
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

        self.latent_dim = latent_dim
        self.n_inducing_points = n_inducing_points
        self.kernel = kernel

        self.encoder = linear_sequential(input_dims=[self.input_dim],
                                         hidden_dims=self.hidden_dims,
                                         output_dim=self.latent_dim,
                                         k_lipschitz=k_lipschitz,
                                         bilipschtiz=bilipschitz,
                                         batch_norm=batch_norm,
                                         p_drop=None
                                         )

        initial_inducing_points, initial_lengthscale = initial_values(
            env, self.encoder, self.n_inducing_points
        )

        self.gp = GP(
            num_outputs=self.output_dim,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel,
        )

        self.net = DKL(self.encoder, self.gp)

    def forward(self, x, task_indices=None, output_type="mean-var"):
        if output_type == "mean-var":
            #dist = self.net(x.float())
            dist = self.net(x.float(), task_indices=task_indices)
            return dist.mean.squeeze(1), dist.variance.squeeze(1)
        elif output_type == "dist":
            dist = self.net(x.float(), task_indices=task_indices)
            return dist
        else:
            raise NotImplementedError


### Taken from DKL repo ###

def initial_values(env, feature_extractor, n_inducing_points):
    # Collect state to init inducing points
    z = []
    n_steps = 1000
    done = True
    feature_extractor.eval()
    with torch.no_grad():
        for step in range(n_steps):
            if done:
                x = env.reset()

            x = torch.tensor([x])#.unsqueeze(0) TODO: check if that is good to comment
            if torch.cuda.is_available():
                x = x.cuda()
                feature_extractor = feature_extractor.cuda()

            z.append(feature_extractor(x.float()).cpu())

            action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            x = new_state
    feature_extractor.train()

    z = torch.cat(z).squeeze()

    initial_inducing_points = _get_initial_inducing_points(
        z.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(z)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()


class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        kernel="RBF",
    ):
        initial_inducing_points = initial_inducing_points.unsqueeze(0).repeat(num_outputs, 1, 1)  # [num_outputs, n_inducing_points, latent_dim] One set of inducing points per task
        n_inducing_points = initial_inducing_points.shape[1]

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            raise ValueError("There is only one possible action")

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, initial_inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_outputs,
        )

        super().__init__(variational_strategy)

        if kernel == "RBF":
            kernel = RBFKernel(batch_shape=batch_shape)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, batch_shape=batch_shape)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, batch_shape=batch_shape)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, batch_shape=batch_shape)
        elif kernel == "RQ":
            kernel = RQKernel(batch_shape=batch_shape)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


class DKL(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does some magic
        on the forward method which is not compatible with a feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.gp = gp

    def forward(self, x, task_indices=None):
        features = self.feature_extractor(x)
        y = self.gp(features, task_indices=task_indices)
        return y
