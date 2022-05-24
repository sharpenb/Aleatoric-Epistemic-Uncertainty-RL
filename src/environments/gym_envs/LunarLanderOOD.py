"""
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.

To see a heuristic landing, run:

python gym/envs/box2d/lunar_lander.py

To play yourself, run:

python examples/agents/keyboard_agent.py LunarLander-v2

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
import random
import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50



class LunarLanderOOD(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, max_episode_steps, ood_noise_magnitude=1.):

        self.ood_noise = ood_noise_magnitude

        self._clock = 0
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(4)

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
        self.state = self.ood_noise * np.random.randn(8)
        obs = np.array(self.state)

        return obs, reward, done, {}

    def reset(self):
        self._clock = 0

        self.state = self.ood_noise * np.random.randn(8)
        return np.array(self.state)

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
