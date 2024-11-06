"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import os
import jax
import jax.numpy as jnp
import numpy as np

import chex
import flax
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal
from wrappers import LogWrapper
from typing import Dict
import popjym
import flashbax as fbx
from model.resnet import ResNet18, ResNet34
import pickle
import popjym
import wandb

from matplotlib import pyplot as plt


class QNetwork(nn.Module):
    action_dim: int

    # def setup(self):
        # self.resnet = ResNet18()
        # using 4 conv layers instead of resnet
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if x.ndim == 3:
            x = jnp.expand_dims(x, axis=0)
        print(f"input: {x.shape}")
        # x = self.resnet(x)
        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output from the convolutional layers
        x = x.reshape((x.shape[0], -1))

        print(f"after:{x.shape}")
        x = nn.Dense(512)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"] # 1e6 // 64 = 15625

    basic_env, env_params = popjym.make(config["ENV_NAME"])
    env_render = popjym.make_render(config["ENV_RENDER"])
    # env = FlattenObservationWrapper(basic_env)

    env = LogWrapper(basic_env)
    # env = basic_env
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)
        init_obs = jax.vmap(env_render.render)(env_state.env_state)
        

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _state, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        # print(_state)
        _obs = env_render.render(_state.env_state)
        # print(_obs.shape)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        # resnet18 = ResNet18(n_classes=512)
        network = QNetwork(action_dim=env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((256,256,3))
        network_params = network.init(_rng, init_x)
        # jax.debug.print("network_params: {}", network_params)
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state
            # jax.debug.print("last_obs: {}", last_obs.shape)
            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(
                rng_a, q_vals, train_state.timesteps
            )  # explore with epsilon greedy_exploration
            # jax.debug.print("q_vals: {}", q_vals)
            # jax.debug.print("action: {}", action)
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            obs = jax.vmap(env_render.render)(env_state.env_state)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, learn_batch.second.obs
                )  # (batch_size, num_actions)
                # q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                # instead of using max, we use the expected value of the next state
                q_next_target = jnp.mean(q_next_target, axis=-1)
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )
                """
                learn_batch = buffer.sample(buffer_state, rng).experience

                # using online network to evaluate next actions 
                q_next_online = network.apply(train_state.params, learn_batch.second.obs)
                next_actions = jnp.argmax(q_next_online, axis=-1)

                # using target network to evaluate q values of next states
                q_next_target = network.apply(train_state.target_network_params, learn_batch.second.obs)
                q_next_target = jnp.take_along_axis(q_next_target, next_actions[:, None], axis=-1).squeeze(-1)

                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )
                """
                def _loss_fn(params):
                    q_vals = network.apply(
                        params, learn_batch.first.obs
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )
            # jax.debug.print("is_learn_time? {}", is_learn_time)
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0)),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
                # "params": train_state.params,
                # "returns": info["returned_episode_returns"][info["returned_episode"]]
            }
            jax.debug.print("metrics: {}", metrics)
            # wandb.watch(network, log="all")
            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)
            # print(f"obs: {obs.shape}")
            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main():

    config = {
        "NUM_ENVS": 10,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 256,
        "TOTAL_TIMESTEPS": 1e6,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 1e-5,
        "LEARNING_STARTS": 100000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "CartPole",
        "ENV_RENDER": "CartPoleRender",
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",  # set to online to activate wandb
        "ENTITY": "",
        "PROJECT": "popjym-acrade-test2",
    }
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'purejaxrl_dqn_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    train_state = {"q_network": outs["runner_state"][0].params}
    with open("./popjym/dqn_flax/train_state.pkl", "wb") as f:
        pickle.dump(train_state, f)



    # with open("./popjym/dqn_flax/train_state.pkl", "rb") as f:
    #     trainstate = pickle.load(f)
    # # print(trainstate)
    # def dqn_evaluate(key: jax.random.PRNGKey, model: nn.Module, config: Dict, params):
    #     rng_reset = jax.random.split(key, 2)
    #     basic_env, env_params = popjym.make(config["ENV_NAME"])
    #     env = LogWrapper(basic_env)
    #     obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(rng_reset, env_params)
    #     env_render = popjym.make_render(config["ENV_RENDER"])
    #     obs = jax.vmap(env_render.render)(env_state.env_state)
    #     plt.axis('off')
    #     plt.imshow(obs[0])
    #     plt.show()
    #     for i in range(50):
    #         rng, rng_act = jax.random.split(key)
    #         q_vals = model.apply(params, obs)
    #         print(q_vals)
    #         action = jnp.argmax(q_vals, axis=-1)
    #         print(action)
    #         rng_step = jax.random.split(rng, 2)
    #         obs2, new_state, reward, term, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
    #         env_state = new_state
    #         obs = jax.vmap(env_render.render)(new_state.env_state)
    #         plt.axis('off')
    #         plt.imshow(obs[0])
    #         plt.show()


    # network = QNetwork(action_dim=5)
    # _rng = jax.random.PRNGKey(0)
    # init_x = jnp.zeros((256,256,3))
    # network_params = network.init(_rng, init_x)
    # network_params = trainstate["q_network"]

    # dqn_evaluate(jax.random.PRNGKey(0), network, config, network_params)


if __name__ == "__main__":
    main()