import logging
import math
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


class CartPoleOOD(gym.Env):
    """
        Description:
            A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum
            starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
        Source:
            This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
        Observation:
            Type: Box(4)
            Num	Observation               Min             Max
            0	Cart Position             -4.8            4.8
            1	Cart Velocity             -Inf            Inf
            2	Pole Angle                -24 deg         24 deg
            3	Pole Velocity At Tip      -Inf            Inf
        Actions:
            Type: Discrete(2)
            Num	Action
            0	Push cart to the left
            1	Push cart to the right
            Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
            pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
            cart underneath it
        Reward:
            Reward is 1 for every step taken, including the termination step
        Starting State:
            All observations are assigned a uniform random value in [-0.05..0.05]
        Episode Termination:
            Pole Angle is more than 12 degrees
            Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
            Episode length is greater than 200
            Solved Requirements
            Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
        """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, max_episode_steps, ood_noise_magnitude=1.):
        self.__version__ = "0.1.0"

        self.ood_noise = ood_noise_magnitude

        self._clock = 0
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(2)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.random_steps = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self._clock += 1
        done = True if self._clock >= self.max_episode_steps else False
        reward = 0.0
        self.state = self.ood_noise * np.random.randn(4)
        obs = np.array(self.state)

        return obs, reward, done, {}

    def reset(self):
        self._clock = 0

        self.state = self.ood_noise * np.random.randn(4)
        return np.array(self.state)

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
