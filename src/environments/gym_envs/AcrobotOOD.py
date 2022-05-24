"""
Classic acrobot task
Original source: https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py
Modified by Mohamad H. Danesh to include wind, and cart friction
"""
import random

import numpy as np
from numpy import sin, cos, pi

from gym import core, spaces
from gym.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How", "Mohamad H. Danesh"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"


class AcrobotOOD(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    def __init__(self, max_episode_steps, ood_noise_magnitude=1.):

        self.ood_noise = ood_noise_magnitude

        self._clock = 0
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(3)

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
        self.state = self.ood_noise * np.random.randn(6)
        obs = np.array(self.state)

        return obs, reward, done, {}

    def reset(self):
        self._clock = 0

        self.state = self.ood_noise * np.random.randn(6)
        return np.array(self.state)

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
