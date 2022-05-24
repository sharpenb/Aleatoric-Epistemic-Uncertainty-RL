import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import os
from sacred import Experiment
import seml

from src.policies.dqn.DQN import DQNLightning

ex = Experiment()
seml.setup_logger(ex)

project_name = 'uncertainty-rl'

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(logger_name,

        # Dataset parameters,
        env_name,
        state_shift,
        action_shift,
        transition_shift,
        reward_shift,
        init_shift,

        # Model parameters
        directory_model,
        model_name,
        checkpoint_name,

        # Testing parameters
        exploration_strategy_name,
        ):

    AVAIL_GPUS = min(1, torch.cuda.device_count())

    #################
    ## Train model ##
    #################
    # Load & log params with W&B
    model_path = f"{directory_model}/{model_name}/{checkpoint_name}"
    random_name = str(random.randint(0, 1e9))
    results_path = f"{directory_model}/eval-{random_name}"
    while os.path.exists(results_path):
        random_name = str(random.randint(0, 1e9))
        results_path = f"{directory_model}/eval-{random_name}"
    os.makedirs(results_path)

    wandb_logger = WandbLogger(save_dir=results_path,
                               name=f'{logger_name}-{model_name}-{checkpoint_name}',
                               project=project_name,
                               log_model='all')

    # Load model
    model = DQNLightning.load_from_checkpoint(model_path)
    wandb_logger.experiment.config.update(model.hparams)

    trainer = pl.Trainer(
        callbacks=[],
        gpus=AVAIL_GPUS,
        max_steps=0,
        logger=wandb_logger
    )

    ################
    ## Test model ##
    ################

    # Testing models checkpointed during training
    model.set_postfix(postfix="")
    model.set_test_env(env_name,
                       state_shift=state_shift,
                       action_shift=action_shift,
                       transition_shift=transition_shift,
                       reward_shift=reward_shift,
                       init_shift=init_shift,
                       )
    model.set_test_decision_strategy(exploration_strategy_name, test_epsilon=0.)
    results = trainer.test(model)[0]

    fail_trace = {
        'fail_trace': seml.evaluation.get_results,
    }

    return {**results, **fail_trace}
