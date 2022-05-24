import time
import random
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import os
from sacred import Experiment
import seml

from src.policies.dqn.DQN import DQNLightning

ex = Experiment()
seml.setup_logger(ex)

project_name = 'uncertainty-rl'
# logger_name = 'dkl'

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
        seed_env,
        env_name,
        ood_env_names,

        # Model parameters
        directory_model,
        encoder_architecture_name,
        hidden_dims,
        k_lipschitz,
        bilipschitz,
        batch_norm,
        uncertainty_architecture_name,
        latent_dim,
        n_inducing_points,
        kernel,

        # Training parameters
        batch_size,
        lr,
        sync_rate,
        tau,
        loss_name,
        gamma,
        reg,
        replay_size,
        warm_start_size,
        exploration_strategy_name,
        eps_last_frame,
        eps_start,
        eps_end,
        episode_length,
        n_test_episodes,
        warm_start_steps,
        n_saved_models,
        max_steps,
        ):

    AVAIL_GPUS = min(1, torch.cuda.device_count())

    #################
    ## Train model ##
    #################
    params_dict= {
        'env_name': env_name,
        'seed_env': seed_env,
        'encoder_architecture_name': encoder_architecture_name,
        'hidden_dims': hidden_dims,
        'k_lipschitz': k_lipschitz,
        'bilipschitz': bilipschitz,
        'batch_norm': batch_norm,
        'uncertainty_architecture_name': uncertainty_architecture_name,
        'uncertainty_params_dict': {'latent_dim': latent_dim, 'n_inducing_points': n_inducing_points, 'kernel': kernel},
        "batch_size": batch_size,
        "lr": lr,
        "sync_rate": sync_rate,
        "tau": tau,
        "loss_name": loss_name,
        "gamma": gamma,
        "reg": reg,
        "replay_size": replay_size,
        "warm_start_size": warm_start_size,
        "exploration_strategy_name": exploration_strategy_name,
        "eps_last_frame": eps_last_frame,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "episode_length": episode_length,
        "n_test_episodes": n_test_episodes,
        "warm_start_steps": warm_start_steps,
        "replay_size": replay_size,
                  }
    model = DQNLightning(**params_dict)

    random_name = str(random.randint(0, 1e6))
    model_path = f"{directory_model}/dqn-{random_name}"
    while os.path.exists(model_path):
        random_name = str(random.randint(0, 1e6))
        model_path = f"{directory_model}/dqn-{random_name}"
    os.makedirs(model_path)

    regular_checkpoint_callback = ModelCheckpoint(monitor="step",
                                                  save_top_k=-1,
                                                  mode="max",
                                                  every_n_train_steps=int(max_steps / n_saved_models),
                                                  dirpath=model_path,
                                                  filename='model-{epoch:02d}')
    wandb_logger = WandbLogger(save_dir=model_path,
                               name=f'{logger_name}-{random_name}',
                               project=project_name,
                               log_model='all')
    wandb_logger.experiment.config.update({"n_test_episodes": n_test_episodes, "n_saved_models": n_saved_models, "max_steps": max_steps})
    t0 = time.time()
    trainer = pl.Trainer(
        callbacks=[regular_checkpoint_callback],
        gpus=AVAIL_GPUS,
        max_steps=max_steps,
        logger=wandb_logger
    )
    trainer.fit(model)
    t1 = time.time()

    ################
    ## Test model ##
    ################

    # Testing models checkpointed during training
    results = {}
    for i, (k_model_path, epoch) in enumerate(regular_checkpoint_callback.best_k_models.items()):
        model = model.load_from_checkpoint(k_model_path)
        model.set_postfix(postfix=str(i))
        model.set_test_env(env_name)

        id_results = trainer.test(model)[0]
        results = {**results, **id_results}
        for ood_env_name in ood_env_names:
            model.set_test_env(ood_env_name)
            ood_results = trainer.test(model)[0]
            results = {**results, **ood_results}


    fail_trace = {
        'training_time': t1 - t0,
        'model_path': model_path,
        'fail_trace': seml.evaluation.get_results,
    }

    wandb.log(
        {"training_time": t1 - t0,
         "model_path": model_path}
              )

    return {**results, **fail_trace}
