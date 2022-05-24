import logging
import math
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


class CartPoleShift(gym.Env):
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

    def __init__(self, max_episode_steps, state_shift=0., action_shift=0., transition_shift=0., reward_shift=0., init_shift=0., when_anomaly_starts=None):
        self.__version__ = "0.1.0"

        self.state_shift = state_shift
        self.action_shift = action_shift
        self.transition_shift = transition_shift
        self.reward_shift = reward_shift
        self.init_shift = init_shift

        self.gravity = (1. + np.random.uniform(low=-self.transition_shift, high=self.transition_shift)) * 9.8
        # self.gravity = np.clip((1. + self.transition_shift * np.random.randn()) * 9.8, a_min=, a_max=)
        self.masscart = (1. + np.random.uniform(low=-self.transition_shift, high=self.transition_shift)) * 1.0
        self.masspole = (1. + np.random.uniform(low=-self.transition_shift, high=self.transition_shift)) * 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = (1. + np.random.uniform(low=-self.transition_shift, high=self.transition_shift)) * 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = (1. + np.random.uniform(low=-self.transition_shift, high=self.transition_shift)) * 10.0
        self._clock = 0
        self.max_episode_steps = max_episode_steps
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.random_steps = []
        self.when_anomaly_starts = when_anomaly_starts
        # TODO: Remove ?
        self.selected_sensors = np.zeros(self.observation_space.shape[0])
        self.selected_sensors[: int(len(self.selected_sensors) / 3)] = 1
        np.random.shuffle(self.selected_sensors)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        is_random = 0
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self._clock += 1

        # Action
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else - self.force_mag
        force = (1 + self.action_shift * np.random.randn()) * force

        # Transition
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)
        if not done:
            done = True if self._clock >= self.max_episode_steps else False

        # Reward
        if not done:
            reward = (1. + self.reward_shift * np.random.randn()) * 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = (1. + self.reward_shift * np.random.randn()) * 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        # State
        obs = np.array(self.state)
        if self._clock > self.when_anomaly_starts:
            obs = (1. + self.state_shift * np.random.randn()) * obs
            is_random = 1
        self.random_steps.append(is_random)
        return obs, reward, done, {}

    def reset(self):
        self._clock = 0

        # Init
        self.state = (1. + self.init_shift * np.random.randn()) * self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.random_steps = []
        # TODO: Remove ?
        self.selected_sensors = np.zeros(self.observation_space.shape[0])
        self.selected_sensors[: int(len(self.selected_sensors) / 3)] = 1
        np.random.shuffle(self.selected_sensors)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None