# import sys
# print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/mc45189/popjym'])

from typing import Any, Tuple, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from jaxtyping import PRNGKeyArray
import chex

import optax
import equinox as eqx
import equinox.nn as nn
import popjym
from popjym.wrappers import LogWrapper
from popjym.model.memorax_equinox import LRU, ResidualModel, add_batch_dim
from popjym.model.memorax_flax.utils import debug_shape
from matplotlib import pyplot as plt
import dataclasses
import wandb
import time
import copy

def debug_shape(x):
    import equinox as eqx
    return eqx.tree_pprint(jax.tree.map(lambda x: {x.shape: x.dtype}, x))

@eqx.filter_jit
def filter_scan(f, init, xs, *args, **kwargs):
    """Same as lax.scan, but allows to have eqx.Module in carry"""
    init_dynamic_carry, static_carry = eqx.partition(init, eqx.is_array)

    def to_scan(dynamic_carry, x):
        carry = eqx.combine(dynamic_carry, static_carry)
        new_carry, out = f(carry, x)
        dynamic_new_carry, _ = eqx.partition(new_carry, eqx.is_array)
        return dynamic_new_carry, out

    out_carry, out_ys = lax.scan(to_scan, init_dynamic_carry, xs, *args, **kwargs)
    return eqx.combine(out_carry, static_carry), out_ys

class QNetwork(eqx.Module):
    """CNN + MLP"""
    action_dim: int
    cnn: nn.Sequential
    rnn: eqx.Module
    trunk: nn.Sequential

    def __init__(self, action_dim: int, key: PRNGKeyArray):
        self.action_dim = action_dim
        keys = jax.random.split(key, 8)
        self.cnn = nn.Sequential([
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, key=keys[0]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, key=keys[1]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, key=keys[2]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, key=keys[3]),
            nn.Lambda(jax.nn.leaky_relu),
        ])
        make_layer_fn = lambda recurrent_size, key: LRU(
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
            key=key,
        )
        self.rnn = ResidualModel(
            make_layer_fn=make_layer_fn,
            input_size=517,
            recurrent_size=512,
            output_size=256,
            num_layers=1,
            key=keys[7],
        )
        self.trunk = nn.Sequential([
            # nn.Linear(in_features=512, out_features=256, key=keys[4]),
            # nn.LayerNorm(shape=256),
            # nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=256, out_features=256, key=keys[5]),
            nn.LayerNorm(shape=256),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=256, out_features=self.action_dim, key=keys[6])
        ])

    def __call__(self, hidden_state, x, done, last_action):
        x = x.transpose((0, 1, 4, 2, 3))
        x = eqx.filter_vmap(eqx.filter_vmap(self.cnn))(x)

        x = x.reshape((x.shape[0], x.shape[1], -1))

        last_action = jax.nn.one_hot(last_action, self.action_dim)
        x = jnp.concatenate([x, last_action], axis=-1)
        rnn_in = (x, done)

        hidden_state, x = eqx.filter_vmap(self.rnn, in_axes=(0, 1), out_axes=(0, 1))(hidden_state, rnn_in)
        hidden_state = eqx.filter_vmap(self.rnn.latest_recurrent_state, in_axes=0)(hidden_state)
        
        q_vals = eqx.filter_vmap(eqx.filter_vmap(self.trunk))(x)

        return hidden_state, q_vals
    
    def initialize_carry(self, key:PRNGKeyArray):
        key_init = jax.random.split(key, 1)
        hidden_state = eqx.filter_jit(self.rnn.initialize_carry)(key=key_init[0])
        return hidden_state

class BoltzmannActor(eqx.Module):
    """Actor that follows a boltzmann (softmax) policy based on the Q values"""
    model: QNetwork

    def __call__(self, x: jax.Array, temperature: jax.Array, key: jax.Array):
        logits = self.model(x)
        # normalize logits, automatic temperature scaling
        logits = logits / (1e-7 + jnp.std(logits))
        return jax.random.categorical(key, logits / temperature)

class Transition(NamedTuple):
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_vals: chex.Array
    infos: chex.Array

class State(eqx.Module):
    """Base class"""
    def replace(self, **kwargs):
        """Replaces existing fields.
        
        E.g., s = State(bork=1, dork=2)
        s.replace(dork=3)
        print(s)
            >> State(bork=1, dork=3)
        """
        fields = self.__dataclass_fields__
        assert set(kwargs.keys()).issubset(fields)
        new_pytree = {}
        for k in fields:
            if k in kwargs:
                new_pytree[k] = kwargs[k]
            else: 
                new_pytree[k] = getattr(self, k)
        return type(self)(**new_pytree)

class TrainState(State):
    model: eqx.Module
    opt:  optax.GradientTransformation
    opt_state: optax.OptState
    timesteps: jnp.ndarray
    n_updates: jnp.ndarray
    grad_steps: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"


    env, env_params = popjym.make(config["ENV_NAME"], partial_observable=config["PARTIAL"])
    env = LogWrapper(env)
    config["TEST_NUM_STEPS"] = config.get(
        "TEST_NUM_STEPS", env_params.max_steps_in_episode
    )
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)
    
    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    def train(rng):
        original_rng = rng[0]
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        rng, _rng, rng_init = jax.random.split(rng, 3)

        network = QNetwork(5, rng)
        
        hidden_state = network.initialize_carry(key=rng_init)
        hidden_state = add_batch_dim(hidden_state, config["NUM_ENVS"])
        
        opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        rng, _rng = jax.random.split(rng)
        opt_state = opt.init(eqx.filter(network, eqx.is_array))
        train_state = TrainState(
            model=network,
            opt=opt,
            opt_state=opt_state,
            timesteps=jnp.array(0),
            n_updates=jnp.array(0),
            grad_steps=jnp.array(0),
        )

 
        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, memory_transitions, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(runner_state, _):
                train_state, memory_transitions, expl_state, test_metrics, rng = runner_state
                hs, last_obs, last_done, last_action, env_state = expl_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

                new_hs, q_vals = train_state.model(
                    hs,
                    _obs,
                    _done,
                    _last_action,
                ) # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = eqx.filter_vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                transition = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1)*reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_vals=q_vals,
                    infos=info,
                )
                new_expl_state = (new_hs, new_obs, new_done, new_action, new_env_state)
                runner_state = (train_state, memory_transitions, new_expl_state, test_metrics, rng)
                return runner_state, transition

            # step the env
            rng, _rng = jax.random.split(rng)
            runner_state, transitions = filter_scan(
                _step_env,
                runner_state,
                None,
                config["NUM_STEPS"],
            )
            train_state, memory_transitions, expl_state, test_metrics, rng = runner_state
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # insert the transitions into the memory
            memory_transitions = jax.tree.map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                update_state, rng = carry
                train_state, memory_transitions = update_state

                def _learn_phase(carry, minibatch):

                    # minibatch shape: num_steps, batch_size, ...
                    # with batch_size = num_envs/num_minibatches

                    train_state, rng = carry
                    # hs = minibatch.last_hs[0]  # hs of oldest step (batch_size, hidden_size)
                    hs = jax.tree.map(lambda hs: hs[0], minibatch.last_hs)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )
                    def _compute_targets(last_q, q_vals, reward, done):
                        def _get_target(lambda_returns_and_next_q, rew_q_done):
                            reward, q, done = rew_q_done
                            lambda_returns, next_q = lambda_returns_and_next_q
                            target_bootstrap = (
                                reward + config["GAMMA"] * (1 - done) * next_q
                            )
                            delta = lambda_returns - next_q
                            lambda_returns = (
                                target_bootstrap
                                + config["GAMMA"] * config["LAMBDA"] * delta
                            )
                            lambda_returns = (1 - done) * lambda_returns + done * reward
                            next_q = jnp.max(q, axis=-1)
                            return (lambda_returns, next_q), lambda_returns

                        lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                        last_q = jnp.max(q_vals[-1], axis=-1)
                        _, targets = jax.lax.scan(
                            _get_target,
                            (lambda_returns, last_q),
                            jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
                            reverse=True,
                        )
                        targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                        return targets

                    def _loss_fn(network):
                        # hs = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), hs)
                        hidden_state, q_vals = network(
                            hs,
                            *agent_in,
                        )  # (num_steps, batch_size, num_actions)
                        # lambda returns are computed using NUM_STEPS as the horizon, and optimizing from t=0 to NUM_STEPS-1
                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        last_q = target_q_vals[-1].max(axis=-1)
                        target = _compute_targets(
                            last_q,  # q_vals at t=NUM_STEPS-1
                            target_q_vals[:-1],
                            minibatch.reward[:-1],
                            minibatch.done[:-1],
                        ).reshape(
                            -1
                        )  # (num_steps-1*batch_size,)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1) # (num_steps, num_agents, batch_size,)
                        chosen_action_qvals = chosen_action_qvals[:-1].reshape(-1)  # (num_steps-1*batch_size,)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, chosen_action_qvals

                    (loss, qvals), grads = eqx.filter_value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.model)
                    updates, new_opt_state = train_state.opt.update(
                        grads, train_state.opt_state, eqx.filter(train_state.model, eqx.is_array), #TODO is_inexact_array
                    )
                    new_network = eqx.apply_updates(train_state.model, updates)
                    new_train_state = train_state.replace(
                        model = new_network,
                        opt_state=new_opt_state,
                        grad_steps=train_state.grad_steps + 1,
                    )
                    return (new_train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    # x: (num_steps, num_envs, ...)
                    x = jax.random.permutation(
                        rng, x, axis=1
                    )  # shuffle the transitions
                    x = x.reshape(
                        x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
                    )  # num_steps, minibatches, batch_size/num_minbatches,
                    x = jnp.swapaxes(x, 0, 1)  # (minibatches, num_steps, batch_size/num_minbatches, ...)
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    memory_transitions,
                )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = filter_scan(
                    _learn_phase, (train_state, rng), minibatches
                )

                update_state = (train_state, memory_transitions)
                return (update_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            update_state = (train_state, memory_transitions)
            (update_state, rng), (loss, qvals) = filter_scan(
                _learn_epoch, (update_state, rng), None, config["NUM_EPOCHS"]
            )
            train_state, memory_transitions = update_state
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update({k: v.mean() for k, v in transitions.infos.items()})
            jax.debug.print("Metrics: {}", metrics)
            jax.debug.print("Infos: {}", transitions.infos)
            
            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_rng)
                jax.debug.print("Timesteps: {}, Returns: {}", metrics['env_step'], transitions.infos['returned_episode_returns'].mean())

            runner_state = (train_state, memory_transitions, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(train_state, rng):
            
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(carry, _):
                train_state, step_state = carry
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
                new_hs, q_vals =train_state.model(
                    hs,
                    _obs,
                    _done,
                    _last_action,
                ) # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(rng_s, env_state, new_action)
                step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
                carry = (train_state, step_state)
                return carry, info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)
            init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
            # initialise_carry_fn = partial(network.apply, method="initialize_carry", mutable=["batch_stats"])
            # init_hs = initialise_carry_fn({"params":train_state.params})
            # init_hs = init_hs[0]
            init_hs = train_state.model.initialize_carry(key=_rng)
            init_hs = add_batch_dim(init_hs, config["TEST_NUM_ENVS"])

            step_state = (
                init_hs,
                init_obs,
                init_done,
                init_action,
                env_state,
                _rng,
            )
            carry = (train_state, step_state)
            carry, infos = filter_scan(
                _greedy_env_step, carry, None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)
        init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
        # initialise_carry_fn = partial(network.apply, method="initialize_carry", mutable=["batch_stats"])
        # init_hs = initialise_carry_fn({"params":train_state.params})

        # init_hs = init_hs[0]
        expl_state = (hidden_state, obs, init_dones, init_action, env_state)
        # expl_state = add_batch_dim(expl_state, 1, 1)

        # step randomly to have the initial memory window
        def _random_step(carry, _):
            _carry, rng = carry
            train_state, expl_state = _carry
            hs, last_obs, last_done, last_action, env_state = expl_state
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
            _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
            _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
            new_hs, q_vals = train_state.model(
                hs,
                _obs,
                _done,
                _last_action,
            ) # (num_envs, hidden_size), (1, num_envs, num_actions)
            q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.) # random actions
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            new_obs, new_env_state, reward, new_done, info = vmap_step(
                config["NUM_ENVS"]
            )(rng_s, env_state, new_action)
            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1)*reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_vals=q_vals,
                infos=info,
            )
            new_expl_state = (new_hs, new_obs, new_done, new_action, new_env_state)
            _carry = (train_state, new_expl_state)
            carry = (_carry, rng)
            return carry, transition
        
        rng, _rng = jax.random.split(rng)
        _carry = (train_state, expl_state)
        (_carry, rng), memory_transitions = filter_scan(
            _random_step,
            (_carry, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        train_state, expl_state = _carry
        expl_state = tuple(expl_state)
        
        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)

        runner_state, metrics = filter_scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train



def evaluate(model):
    seed = jax.random.PRNGKey(10)
    env, env_params = popjym.make("CartPoleHard")
    obs, state = env.reset(seed, env_params)
    plt.axis('off')
    plt.imshow(obs)
    plt.show()
    for i in range(20):
        rng, rng_act, rng_step, _rng = jax.random.split(seed, 4)
        actor_mean, value = model(obs)
        action = jnp.argmax(actor_mean)
        obs2, new_state, reward, term, _ = env.step(rng_step, state, action=action)
        state = new_state
        plt.axis('off')
        plt.imshow(obs)
        plt.show()

def single_run(config):

    # config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn_rnn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'pqn_rnn_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = eqx.filter_jit(eqx.filter_vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    # if config.get("SAVE_PATH", None) is not None:
    #     from jaxmarl.wrappers.baselines import save_params

    #     model_state = outs["runner_state"][0]
    #     save_dir = os.path.join(config["SAVE_PATH"], env_name)
    #     os.makedirs(save_dir, exist_ok=True)
    #     OmegaConf.save(
    #         config,
    #         os.path.join(
    #             save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
    #         ),
    #     )

    #     for i, rng in enumerate(rngs):
    #         params = jax.tree_map(lambda x: x[i], model_state.params)
    #         save_path = os.path.join(
    #             save_dir,
    #             f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
    #         )
    #         save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "pqn_rnn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


# @hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    # config = OmegaConf.to_container(config)
    # print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)



if __name__ == "__main__":
    config = {
    "TOTAL_TIMESTEPS": 1e7,
    "TOTAL_TIMESTEPS_DECAY": 1e7, # will be used for decay functions, in case you want to test for less timesteps and keep decays same
    "NUM_ENVS": 16, # parallel environments
    "MEMORY_WINDOW": 4, # steps of previous episode added in the rnn training horizon
    "NUM_STEPS": 128, # steps per environment in each update
    "EPS_START": 1,
    "EPS_FINISH": 0.2,
    "EPS_DECAY": 0.2, # ratio of total updates
    "NUM_MINIBATCHES": 16, # minibatches per epoch
    "NUM_EPOCHS": 4, # minibatches per epoch
    "NORM_INPUT": False,
    "HIDDEN_SIZE": 256,
    "NUM_LAYERS": 2,
    "NORM_TYPE": "layer_norm", # layer_norm or batch_norm
    "LR": 0.0001,
    "MAX_GRAD_NORM": 10,
    "LR_LINEAR_DECAY": True,
    "REW_SCALE": 0.1,
    "GAMMA": 0.99,
    "LAMBDA": 0.95,
    "HYP_TUNE": False,
    "ENTITY": "",
    "PROJECT": "equinox_pqn_lru_clean",
    "WANDB_MODE": "disabled",
    "SEED": 0,
    "NUM_SEEDS": 1,
    "RNN_TYPE": "lru", # type of rnn to use, lru, fart, gilr, gru
    "PARTIAL": True,

    # env specific
    "ENV_NAME": "NavigatorEasy", # should work also for Acrobot-v1 but might need some tuning
    "ENV_KWARGS": {},

    # evaluation
    "TEST_DURING_TRAINING": True ,
    "TEST_INTERVAL": 0.05 ,# in terms of total updates
    "TEST_NUM_ENVS": 128,
    # TEST_NUM_STEPS: 128 # setup automatically with max env timesteps
    "EPS_TEST": 0, # 0 for greedy policy
    }
    main(config)
