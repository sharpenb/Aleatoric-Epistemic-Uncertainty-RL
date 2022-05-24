import gym

gym_environments = env_ids = [env_spec.id for env_spec in gym.envs.registry.all()]

shift_gym_environments = {
    'CartPoleShift-v0': None,
    'AcrobotShift-v0': None,
    'LunarLanderShift-v0': None,
    'PendulumShift-v0': None,
}



def get_environment(env_name, seed_env, state_shift=0., action_shift=0., transition_shift=0., reward_shift=0., init_shift=0., anomalous_time=0):
    if env_name in shift_gym_environments:
        env = gym.make(env_name,
                       when_anomaly_starts=anomalous_time,
                       state_shift=state_shift,
                       action_shift=action_shift,
                       transition_shift=transition_shift,
                       reward_shift=reward_shift,
                       init_shift=init_shift)
    elif env_name in gym_environments:
        env = gym.make(env_name)
    else:
        raise NotImplementedError
    env.seed(seed_env)
    return env
