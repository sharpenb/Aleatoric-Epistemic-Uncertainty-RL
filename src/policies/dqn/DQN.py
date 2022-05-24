from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import gym
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import DistributedType
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.environments.Environment import get_environment
from src.policies.dqn.mlp.Agent import Agent as MLPAgent
from src.policies.dqn.dropout.Agent import Agent as DropOutAgent
from src.policies.dqn.ensemble.Agent import Agent as EnsembleAgent
from src.policies.dqn.dkl.Agent import Agent as DKLAgent
from src.policies.dqn.postnet.Agent import Agent as PostNetAgent
from src.policies.dqn.ReplayBuffer import ReplayBuffer
from src.policies.dqn.RLDataset import RLDataset
from src.metrics.auc_scores import *


agent_dict = {
    "Standard": MLPAgent,
    "DropOut": DropOutAgent,
    "Ensemble": EnsembleAgent,
    "DKL": DKLAgent,
    "PostNet": PostNetAgent,
}


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            batch_size: int = 16,
            lr: float = 1e-2,
            tau: float = 1,
            env_name: str = "CartPole-v0",
            seed_env: int = 0,
            encoder_architecture_name: str = "MLP",
            hidden_dims: int = [128],
            k_lipschitz: str = None,
            bilipschitz: bool = False,
            batch_norm: bool = False,
            uncertainty_architecture_name: str = "None",
            uncertainty_params_dict: dict = None,
            exploration_strategy_name: str = "epsilon-greedy",
            loss_name: str = "MSE",
            gamma: float = 0.99,
            reg: float = 0.,
            sync_rate: int = 10,
            replay_size: int = 1000,
            warm_start_size: int = 1000,
            eps_last_frame: int = 1000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 200,
            n_test_episodes: int = 10,
            warm_start_steps: int = 1000,
            seed_model: int = 0,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env_name: gym environment tag
            seed_env: environment seed
            encoder_architecture_name: name of the encoder architectuer
            uncertainty_architecture_name: name of the uncertainty method
            loss_name: name of the loss
            gamma: discount factor
            reg: regularization factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = get_environment(env_name=self.hparams.env_name,
                                   seed_env=self.hparams.seed_env)
        # self.val_env = get_environment(env_name=self.hparams.env_name,
        #                                seed_env=self.hparams.seed_env)
        self.test_env = None
        self.postfix = ""

        self.replay_buffer = ReplayBuffer(self.hparams.replay_size)

        if len(self.env.observation_space.shape) > 0:
            obs_size = self.env.observation_space.shape[0]
        elif len(self.env.observation_space.shape) == 0:
            obs_size = 1
        n_actions = self.env.action_space.n

        params_dict = {
            "obs_size": obs_size,
            "n_actions": n_actions,
            "encoder_architecture_name": self.hparams.encoder_architecture_name,
            "hidden_dims": self.hparams.hidden_dims,
            "k_lipschitz": self.hparams.k_lipschitz,
            "bilipschitz": self.hparams.bilipschitz,
            "batch_norm": self.hparams.batch_norm,
            "exploration_strategy_name": self.hparams.exploration_strategy_name,
            "loss_name": self.hparams.loss_name,
            "gamma": self.hparams.gamma,
            "reg": self.hparams.reg,
            "env": self.env,
            "seed": self.hparams.seed_model
        }
        if self.hparams.uncertainty_params_dict is not None:  # We use the specified uncertainty parameters instead of the default.
            params_dict = {**params_dict, **self.hparams.uncertainty_params_dict}
        self.agent = agent_dict[self.hparams.uncertainty_architecture_name](**params_dict)

        self.agent.reset(self.env)
        self.total_reward = 0
        self.episode_reward = 0
        self.n_finished_training_episodes = 0

        self.populate(self.hparams.warm_start_steps)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.agent.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ## Environment methods ##
    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences i.e. get train loader"""
        dataset = RLDataset(self.replay_buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        """Initialize an (almot) empty Replay Buffer dataset to fullful pl requirements i.e. get test loader"""
        dataset = RLDataset(self.replay_buffer, 1)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def set_test_env(self, env_name: str,
                     state_shift: float = None,
                     action_shift: float = None,
                     transition_shift: float = None,
                     reward_shift: float = None,
                     init_shift: float = None,
                     ):
        """set a new environment

        Args:
            env_name: environment name to create
        """
        self.hparams.test_env_name = env_name
        if state_shift is None or action_shift is None or transition_shift is None or reward_shift is None or init_shift is None:
            self.test_env = get_environment(env_name=self.hparams.test_env_name,
                                            seed_env=self.hparams.seed_env)
        else:
            self.test_env = get_environment(env_name=self.hparams.test_env_name,
                                        seed_env=self.hparams.seed_env,
                                        state_shift=state_shift,
                                        action_shift=action_shift,
                                        transition_shift=transition_shift,
                                        reward_shift=reward_shift,
                                        init_shift=init_shift)

        self.agent.reset(self.test_env)

        self.total_reward = 0
        self.episode_reward = 0

    def set_test_decision_strategy(self, test_decision_strategy_name: str, test_epsilon: float):
        """set a new environment

        Args:
            env_name: environment name to create
        """
        self.agent.exploration_strategy_name = test_decision_strategy_name
        self.test_epsilon = test_epsilon

    def set_postfix(self, postfix):
        self.postfix = postfix

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            env: environment to use to populate
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(env=self.env, replay_buffer=self.replay_buffer, epsilon=1.0)

    ## Prediction methods ##
    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.agent.net(x)
        return output

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        if self.hparams.exploration_strategy_name == "epsilon-greedy":
            epsilon = max(
                self.hparams.eps_end,
                self.hparams.eps_start - (self.global_step / self.hparams.eps_last_frame),
            )
        else:
            epsilon = self.hparams.eps_start

        # step through environment with agent
        reward, done, _, _ = self.agent.play_step(self.env, self.replay_buffer, epsilon, device)
        self.episode_reward += reward

        # calculates training loss & statistics
        loss = self.agent.compute_loss(batch)
        mean, aleatoric_uncertainty, epistemic_uncertainty = self.agent.compute_prediction_stastistics(batch)

        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.n_finished_training_episodes += 1

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.agent.soft_update(self.hparams.tau)

        log = {
            "training_mean": torch.tensor(mean).to(device),
            "training_aleatoric_uncertainty": torch.tensor(aleatoric_uncertainty).to(device),
            "training_epistemic_uncertainty": torch.tensor(epistemic_uncertainty).to(device),
            "training_total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "training_loss": loss,
            "n_finished_training_episodes": self.n_finished_training_episodes,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "training_total_reward": torch.tensor(self.total_reward).to(device),
        }

        self.log("epsilon", epsilon)
        self.log("training_mean", log["training_mean"])
        self.log("training_aleatoric_ucertainty", log["training_aleatoric_uncertainty"])
        self.log("training_epistemic_uncertainty", log["training_epistemic_uncertainty"])
        self.log("training_total_reward", log["training_total_reward"])
        self.log("reward", log["reward"])
        self.log("training_loss", log["training_loss"])
        self.log("n_finished_training_episodes", log["n_finished_training_episodes"])

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    # def training_epoch_end(self, step_outputs):
    #     if self.current_epoch % (2 * self.hparams.n_test_episodes) == 0:  # TODO: criteria is handcrafted
    #         with torch.no_grad():
    #             results = {
    #                 "val_episode_lengths": [],
    #                 "val_total_rewards": [],
    #                 "val_epistemic_uncertainty": [],
    #                 "val_aleatoric_uncertainty": [],
    #             }
    #             self.agent.reset(self.val_env)
    #             for i in range(int(self.hparams.n_test_episodes / 2)):  # TODO: criteria is handcrafted
    #                 total_reward = 0
    #                 for step in range(self.hparams.episode_length):
    #                     reward, done, epistemic_uncertainty, aleatoric_uncertainty = self.agent.play_step(self.val_env, None, 0.)
    #                     # loss = self.agent.compute_loss(batch) # TODO: add val loss tracking ?
    #                     total_reward += reward
    #                     results["val_epistemic_uncertainty"].append(float(epistemic_uncertainty))
    #                     results["val_aleatoric_uncertainty"].append(float(aleatoric_uncertainty))
    #                     if done:
    #                         results["val_total_rewards"].append(total_reward)
    #                         results["val_episode_lengths"].append(step)
    #                         break
    #             self.log("val_epistemic_uncertainty", np.mean(results["val_epistemic_uncertainty"]))
    #             self.log("val_aleatoric_uncertainty", np.mean(results["val_aleatoric_uncertainty"]))
    #             self.log("val_total_rewards", np.mean(results["val_total_rewards"]))
    #             self.log("val_episode_lengths", np.mean(results["val_episode_lengths"]))

    def test_step(self, batch, batch_idx):
        """Skip test steps which would browse the ReplayBuffer
        """
        return -1

    def test_epoch_end(self, step_outputs):
        postfix = self.postfix
        results = {
            f"train_episode_lengths_{postfix}": [],
            f"train_total_rewards_{postfix}": [],
            f"train_epistemic_uncertainty_{postfix}": [],
            f"train_aleatoric_uncertainty_{postfix}": [],
            f"test_episode_lengths_{postfix}": [],
            f"test_total_rewards_{postfix}": [],
            f"test_epistemic_uncertainty_{postfix}": [],
            f"test_aleatoric_uncertainty_{postfix}": [],
        }

        # Train environment
        if self.hparams.test_env_name != self.hparams.env_name:
            for i in range(self.hparams.n_test_episodes):
                print('ID', i)
                total_reward = 0
                step = 0
                done = False
                while not done and step < self.hparams.episode_length:
                    reward, done, epistemic_uncertainty, aleatoric_uncertainty = self.agent.play_step(self.env, None, 0.)
                    total_reward += reward
                    step += 1
                    results[f"train_epistemic_uncertainty_{postfix}"].append(float(epistemic_uncertainty))
                    results[f"train_aleatoric_uncertainty_{postfix}"].append(float(aleatoric_uncertainty))
                    if done or step >= self.hparams.episode_length:
                        results[f"train_total_rewards_{postfix}"].append(total_reward)
                        results[f"train_episode_lengths_{postfix}"].append(step)

        # Test environment
        for i in range(self.hparams.n_test_episodes):
            print('OOD', i)
            total_reward = 0
            step = 0
            done = False
            while not done and step < self.hparams.episode_length:
                reward, done, epistemic_uncertainty, aleatoric_uncertainty = self.agent.play_step(self.test_env, None, 0.)
                total_reward += reward
                step += 1
                results[f"test_epistemic_uncertainty_{postfix}"].append(float(epistemic_uncertainty))
                results[f"test_aleatoric_uncertainty_{postfix}"].append(float(aleatoric_uncertainty))
                if done or step >= self.hparams.episode_length:
                    results[f"test_total_rewards_{postfix}"].append(total_reward)
                    results[f"test_episode_lengths_{postfix}"].append(step)
                    break

        self.log(f"{self.hparams.test_env_name}-mean_reward_{postfix}", np.mean(results[f"test_total_rewards_{postfix}"]))
        self.log(f"{self.hparams.test_env_name}-std_reward_{postfix}", np.var(results[f"test_total_rewards_{postfix}"])**.5)
        self.log(f"{self.hparams.test_env_name}-mean_epistemic_uncertainty_{postfix}", np.mean(results[f"test_epistemic_uncertainty_{postfix}"]))
        self.log(f"{self.hparams.test_env_name}-std_epistemic_uncertainty_{postfix}", np.var(results[f"test_epistemic_uncertainty_{postfix}"])**.5)
        self.log(f"{self.hparams.test_env_name}-mean_aleatoric_uncertainty_{postfix}", np.mean(results[f"test_aleatoric_uncertainty_{postfix}"]))
        self.log(f"{self.hparams.test_env_name}-std_aleatoric_uncertainty_{postfix}", np.var(results[f"test_aleatoric_uncertainty_{postfix}"])**.5)
        if self.hparams.test_env_name != self.hparams.env_name:
            aleatoric_scores = np.concatenate([results[f"test_aleatoric_uncertainty_{postfix}"], results[f"train_aleatoric_uncertainty_{postfix}"]])
            epistemic_scores = np.concatenate([results[f"test_epistemic_uncertainty_{postfix}"], results[f"train_epistemic_uncertainty_{postfix}"]])
            binary_classes = np.concatenate([np.ones_like(results[f"test_aleatoric_uncertainty_{postfix}"]), np.zeros_like(results[f"train_aleatoric_uncertainty_{postfix}"])])
            self.log(f"{self.hparams.test_env_name}-AUC-ROC-aleatoric_{postfix}", auc_roc(aleatoric_scores, binary_classes))
            self.log(f"{self.hparams.test_env_name}-AUC-PR-in-aleatoric_{postfix}", auc_apr(aleatoric_scores, binary_classes))
            self.log(f"{self.hparams.test_env_name}-AUC-PR-out-aleatoric_{postfix}", auc_apr(- aleatoric_scores, 1 - binary_classes))
            self.log(f"{self.hparams.test_env_name}-AUC-ROC-epistemic_{postfix}", auc_roc(epistemic_scores, binary_classes))
            self.log(f"{self.hparams.test_env_name}-AUC-PR-in-epistemic_{postfix}", auc_apr(epistemic_scores, binary_classes))
            self.log(f"{self.hparams.test_env_name}-AUC-PR-out-epistemic_{postfix}", auc_apr(- epistemic_scores, 1 - binary_classes))

        return results
