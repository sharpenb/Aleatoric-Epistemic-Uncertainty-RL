import logging
from gym.envs.registration import register
import src.environments.gym_envs

logger = logging.getLogger(__name__)

##############
## OOD Envs ##
##############

register(
    id='CartPoleOOD-v0',
    entry_point='src.environments.gym_envs:CartPoleOOD',
    kwargs={'max_episode_steps': 500}
)

register(
    id='AcrobotOOD-v0',
    entry_point='src.environments.gym_envs:AcrobotOOD',
    kwargs={'max_episode_steps': 500}
)

register(
    id='LunarLanderOOD-v0',
    entry_point='src.environments.gym_envs:LunarLanderOOD',
    kwargs={'max_episode_steps': 500}
)

##################
## Shifted Envs ##
##################

register(
    id='CartPoleShift-v0',
    entry_point='src.environments.gym_envs:CartPoleShift',
    kwargs={'max_episode_steps': 500}
)

register(
    id='AcrobotShift-v0',
    entry_point='src.environments.gym_envs:AcrobotShift',
    kwargs={}
)

register(
    id='LunarLanderShift-v0',
    entry_point='src.environments.gym_envs:LunarLanderShift',
    kwargs={}
)
