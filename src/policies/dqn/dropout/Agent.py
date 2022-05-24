from typing import Tuple

import gym
import numpy as np
import torch
from torch import Tensor, nn

from src.policies.dqn.ReplayBuffer import Experience
from src.policies.dqn.ReplayBuffer import ReplayBuffer
from src.policies.dqn.dropout.DropOut import MLPDropOut


class Agent(nn.Module):
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 encoder_architecture_name: str = "MLP",
                 hidden_dims: int = [128],
                 k_lipschitz: str = None,
                 bilipschitz: bool = False,
                 batch_norm: bool = False,
                 p_drop: float = .25,
                 n_samples: int = 20,
                 exploration_strategy_name: str = "epsilon-greedy",
                 loss_name: str = "MSE",
                 gamma: float = .99,
                 reg: float = 0.,
                 env: gym.Env = None,
                 seed: int = 0,) -> None:
        """
        Args:
            obs_size: dimension of observations
            n_actions: dimension of actions
            gamma: discounting factor
            env: training environment
        """
        super().__init__()

        self.encoder_architecture_name = encoder_architecture_name
        if self.encoder_architecture_name == "MLP":
            self.net = MLPDropOut(obs_size=obs_size,
                                  n_actions=n_actions,
                                  hidden_dims=hidden_dims,
                                  k_lipschitz=k_lipschitz,
                                  bilipschitz=bilipschitz,
                                  batch_norm=batch_norm,
                                  p_drop=p_drop,
                                  n_samples=n_samples,
                                  seed=seed)
            self.target_net = MLPDropOut(obs_size=obs_size,
                                         n_actions=n_actions,
                                         hidden_dims=hidden_dims,
                                         k_lipschitz=k_lipschitz,
                                         bilipschitz=bilipschitz,
                                         batch_norm=batch_norm,
                                         p_drop=p_drop,
                                         n_samples=n_samples,
                                         seed=seed)
        else:
            raise NotImplementedError

        self.exploration_strategy_name = exploration_strategy_name
        self.loss_name = loss_name
        self.gamma = gamma
        self.reg = reg

    ## Environments Methods ##
    def reset(self, env: gym.Env) -> None:
        """Resents the environment and updates the state."""
        self.state = env.reset()

    ## Action Methods ##
    def forward(self, x):
        return self.net(x)

    def get_action(self, env: gym.Env, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        state = torch.tensor([self.state])
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if device not in ["cpu"]:
            state = state.cuda(device)

        self.net = self.net.to(device)
        self.target_net = self.target_net.to(device)

        if self.exploration_strategy_name == "epsilon-greedy":
            self.net.eval()
            mean_q_values, var_q_values, epistemic_var_q_values = self.net(state)
            self.net.train()

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                _, action = torch.max(mean_q_values, dim=1)
                action = int(action.item())
        elif self.exploration_strategy_name == "sampling-epistemic":
            self.net.eval()
            means_q_values, vars_q_values = self.net(state, aggregate=False)
            mean_q_values, var_q_values, epistemic_var_q_values = torch.mean(means_q_values, -2), torch.mean(vars_q_values, -2), torch.var(means_q_values, -2)  # Useful to track mean & aleatoric
            member_samples = torch.distributions.Categorical(logits=torch.ones(means_q_values.shape[1])).sample([means_q_values.shape[0]])
            mean_q_values_samples = means_q_values[torch.arange(member_samples.shape[0]), member_samples, :]  # Sample from epistemic (empirical) distribution
            self.net.train()

            _, action = torch.max(mean_q_values_samples, dim=1)
            action = int(action.item())
        elif self.exploration_strategy_name == "sampling-aleatoric":
            self.net.eval()
            mean_q_values, var_q_values, epistemic_var_q_values = self.net(state)
            mean_q_values_samples = torch.distributions.Normal(mean_q_values, var_q_values).rsample([])  # Sample from aleatoric distribution
            self.net.train()

            _, action = torch.max(mean_q_values_samples, dim=1)
            action = int(action.item())
        else:
            raise NotImplementedError

        return action, mean_q_values[:, action], var_q_values[:, action], epistemic_var_q_values[:, action]

    @torch.no_grad()
    def play_step(
            self,
            env: gym.Env = None,
            replay_buffer: ReplayBuffer = None,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action, mean_q_value, var_q_value, epistemic_var_q_value = self.get_action(env, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        if replay_buffer is not None:
            replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset(env)
        return reward, done, epistemic_var_q_value, var_q_value

    ## Loss Methods ##
    def compute_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        if self.loss_name == "MSE":
            return self.mse_loss(batch=batch)
        elif self.loss_name == "SmoothL1":
            return self.smooth_L1_loss(batch=batch)
        elif self.loss_name == "GaussianNLL":
            return self.gaussian_nll_loss(batch=batch)
        else:
            raise NotImplementedError

    def compute_prediction_stastistics(self, batch: Tuple[Tensor, Tensor]):
        """Calculates the statistics of predictions on a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            mean prediction, aleatoric uncertainty, epistemic uncertainty
        """
        states, actions, rewards, dones, next_states = batch

        if len(states.shape) == 1:
            states = states.unsqueeze(-1)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(-1)

        with torch.no_grad():
            mean_state_action_values, var_state_action_values, epistemic_var_state_action_values = self.net(states)
            mean_state_action_values = mean_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            var_state_action_values = var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            epistemic_var_state_action_values = epistemic_var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        return mean_state_action_values.mean(), var_state_action_values.mean(), epistemic_var_state_action_values.mean()

    def mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        if len(states.shape) == 1:
            states = states.unsqueeze(-1)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(-1)

        mean_state_action_values, var_state_action_values, epistemic_var_state_action_values = self.net(states)
        mean_state_action_values = mean_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        var_state_action_values = var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        epistemic_var_state_action_values = epistemic_var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            mean_next_state_values, var_next_state_values, epistemic_var_next_state_values = self.target_net(next_states)
            mean_next_state_values = mean_next_state_values.max(1)[0]
            mean_next_state_values[dones] = 0.0
            mean_next_state_values = mean_next_state_values.detach()

        expected_state_action_values = mean_next_state_values * self.gamma + rewards

        return nn.MSELoss()(mean_state_action_values, expected_state_action_values) + self.reg * nn.MSELoss()(epistemic_var_next_state_values, torch.zeros_like(epistemic_var_next_state_values))

    def smooth_L1_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        if len(states.shape) == 1:
            states = states.unsqueeze(-1)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(-1)

        mean_state_action_values, var_state_action_values, epistemic_var_state_action_values = self.net(states)
        mean_state_action_values = mean_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        var_state_action_values = var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        epistemic_var_state_action_values = epistemic_var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            mean_next_state_values, var_next_state_values, epistemic_var_next_state_values = self.target_net(next_states)
            mean_next_state_values = mean_next_state_values.max(1)[0]
            mean_next_state_values[dones] = 0.0
            mean_next_state_values = mean_next_state_values.detach()

        expected_state_action_values = mean_next_state_values * self.gamma + rewards

        return nn.SmoothL1Loss()(mean_state_action_values, expected_state_action_values) + self.reg * nn.SmoothL1Loss()(epistemic_var_next_state_values, torch.zeros_like(epistemic_var_next_state_values))

    def gaussian_nll_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the Gaussian NLL loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        if len(states.shape) == 1:
            states = states.unsqueeze(-1)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(-1)

        mean_state_action_values, var_state_action_values, epistemic_var_state_action_values = self.net(states)
        mean_state_action_values = mean_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        var_state_action_values = var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        epistemic_var_state_action_values = epistemic_var_state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            mean_next_state_values, var_next_state_values, epistemic_var_next_state_values = self.target_net(next_states)
            mean_next_state_values = mean_next_state_values.max(1)[0]
            mean_next_state_values[dones] = 0.0
            mean_next_state_values = mean_next_state_values.detach()

        expected_state_action_values = mean_next_state_values * self.gamma + rewards

        return nn.GaussianNLLLoss()(mean_state_action_values, expected_state_action_values, var_state_action_values) + self.reg * nn.MSELoss()(epistemic_var_next_state_values, torch.zeros_like(epistemic_var_next_state_values))

    ## Parameter Update Methods ##
    def soft_update(self, tau: float):
        """Perfrom a soft update of the target network parameters.

        Args:
            tau: share of the current network parameters in the soft update
        """
        net_state_dict = self.net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in net_state_dict:
            target_net_state_dict[key] = (1. - tau) * target_net_state_dict[key] + tau * net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

