from typing import List
import gym
import torch
from torch import Tensor, nn
from typing import Tuple

import src.policies.dqn.postnet.distributions as D
from src.policies.dqn.postnet.flow import NormalizingFlow, RadialFlow
from src.policies.dqn.postnet.output import Output, NormalOutput
from src.policies.dqn.postnet.scaler import CertaintyBudget, EvidenceScaler
from src.architectures.linear_sequential import linear_sequential


class MLPPostNet(nn.Module):
    """Drop Out MLP network."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 hidden_dims: int = [128],
                 k_lipschitz: str = None,
                 bilipschitz: bool = False,
                 batch_norm: bool = False,
                 latent_dim: int = 8,
                 flow_length: int = 8,
                 flow_type: str = "radial",
                 prior_mean: float = 0.,
                 prior_scale: float = 10.,
                 prior_evidence: float = 1.,
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
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.encoder = linear_sequential(input_dims=[self.input_dim],
                                         hidden_dims=self.hidden_dims,
                                         output_dim=self.latent_dim,
                                         k_lipschitz=k_lipschitz,
                                         bilipschtiz=bilipschitz,
                                         batch_norm=batch_norm,
                                         p_drop=None
                                         )
        if self.flow_type == "radial":
            flows = [RadialFlow(dim=self.latent_dim, num_layers=self.flow_length) for i in range(self.output_dim)]
        else:
            raise NotImplementedError
        outputs = [NormalOutput(dim=self.latent_dim, prior_mean=prior_mean, prior_scale=prior_scale, prior_evidence=prior_evidence) for i in range(self.output_dim)]

        self.net = NaturalPosteriorNetworkModel(
            latent_dim=self.latent_dim,
            encoder=self.encoder,
            flows=flows,
            outputs=outputs,
            certainty_budget="normal",
        )

    def forward(self, x, task_indices=None, output_type="mean-var-prob"):
        if output_type == "mean-var-prob":
            dist, log_prob = self.net(x.float())
            return dist.maximum_a_posteriori().mean().squeeze(1), dist.maximum_a_posteriori().uncertainty().squeeze(1), log_prob.squeeze(1)
        elif output_type == "dist":
            dist, log_prob = self.net(x.float(), task_indices=task_indices)
            return dist, log_prob.squeeze(1)
        else:
            raise NotImplementedError


### Taken from NatPN repo ###

class NaturalPosteriorNetworkModel(nn.Module):
        """
        Implementation of the NatPN module. This class only describes the forward pass through the
        model and can be compiled via TorchScript.
        """

        def __init__(
                self,
                latent_dim: int,
                encoder: nn.Module,
                flows: List[NormalizingFlow],
                outputs: List[Output],
                certainty_budget: CertaintyBudget = "normal",
        ):
            """
            Args:
                latent_dim: The dimension of the latent space to which the model's encoder maps.
                config: The model's intrinsic configuration.
                encoder: The model's (deep) encoder which maps input to a latent space.
                flow: The model's normalizing flow which yields the evidence of inputs based on their
                    latent representations.
                output: The model's output head which maps each input's latent representation linearly
                    to the parameters of the target distribution.
                certainty_budget: The scaling factor for the certainty budget that the normalizing
                    flow can draw from.
            """
            super().__init__()
            self.encoder = encoder
            self.flows = nn.ModuleList(flows)
            self.outputs = nn.ModuleList(outputs)
            self.scaler = EvidenceScaler(latent_dim, certainty_budget)

        def forward(self, x: torch.Tensor, task_indices=None) -> Tuple[D.Posterior, torch.Tensor]:
            """
            Performs a Bayesian update over the target distribution for each input independently. The
            returned posterior distribution carries all information about the prediction.
            Args:
                x: The inputs that are first passed to the encoder.
            Returns:
                The posterior distribution for every input along with their log-probabilities. The
                same probabilities are returned from :meth:`log_prob`.
            """
            update, log_prob = self.posterior_update(x, task_indices)
            return self.outputs[0].prior.update(update), log_prob  # All prior are the same for different outputs

        def posterior_update(self, x: torch.Tensor, task_indices=None) -> Tuple[D.PosteriorUpdate, torch.Tensor]:
            """
            Computes the posterior update over the target distribution for each input independently.
            Args:
                x: The inputs that are first passed to the encoder.
            Returns:
                The posterior update for every input and the true log-probabilities.
            """
            batch_size, output_dim = x.shape[0], len(self.outputs)

            z = self.encoder.forward(x)
            if z.dim() > 2:
                z = z.permute(0, 2, 3, 1)

            sufficient_statistics = torch.zeros((batch_size, output_dim, 2)).to(z.device.type) # The dim of the suff. stat. for  InverseNormal is 2.
            log_evidence = torch.zeros((batch_size, output_dim)).to(z.device.type)
            log_prob = torch.zeros((batch_size, output_dim)).to(z.device.type)
            for i in range(output_dim):
                prediction = self.outputs[i].forward(z)
                sufficient_statistics[:, i, :] = prediction.expected_sufficient_statistics()

                log_prob[:, i] = self.flows[i].forward(z)
                log_evidence[:, i] = self.scaler.forward(log_prob[:, i])

            if task_indices is not None:
                sufficient_statistics = sufficient_statistics[torch.arange(batch_size), task_indices, :]
                log_evidence = log_evidence[torch.arange(batch_size), task_indices]
                log_prob = log_prob[torch.arange(batch_size), task_indices]

            return D.PosteriorUpdate(sufficient_statistics, log_evidence), log_prob

        def log_prob(self, x: torch.Tensor, track_encoder_gradients: bool = True) -> torch.Tensor:
            """
            Computes the (scaled) log-probability of observing the given inputs.
            Args:
                x: The inputs that are first passed to the encoder.
                track_encoder_gradients: Whether to track the gradients of the encoder.
            Returns:
                The per-input log-probability.
            """
            batch_size, output_dim = x.shape[0], len(self.outputs)

            with torch.set_grad_enabled(self.training and track_encoder_gradients):
                z = self.encoder.forward(x)
                if z.dim() > 2:
                    z = z.permute(0, 2, 3, 1)

                log_prob = torch.zeros((batch_size, output_dim)).to(z.device.type)
                for i in range(output_dim):
                    log_prob[:, i] = self.flows[i].forward(z)

            return log_prob



