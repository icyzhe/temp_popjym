from typing import Any, Tuple
import jax
import jax.numpy as jnp

import optax
import equinox as eqx
from equinox import nn
import popjym
from popjym.wrappers import LogWrapper
import wandb

def debug_shape(x):
    jax.tree.map(
        lambda x: print(x.shape) if hasattr(x, "shape") else x, x
    )

class QNetwork(eqx.Module):
    """CNN + MLP"""
    action_dim: int
    cnn: eqx.Module
    trunk: eqx.Module

    def __init__(self, action_dim: int, key):
        self.action_dim = action_dim
        keys = jax.random.split(key, 7)
        self.cnn = nn.Sequential([
            nn.Conv2d(3, 64, 3, 1, key=keys[0]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.AdaptiveMaxPool2d(128),

            nn.Conv2d(64, 128, 3, 1, key=keys[1]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.AdaptiveMaxPool2d(64),

            nn.Conv2d(128, 256, 3, 1, key=keys[2]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.AdaptiveMaxPool2d(32),

            nn.Conv2d(256, 512, 3, 1, key=keys[3]),
            nn.Lambda(jax.nn.leaky_relu),
            nn.AdaptiveMaxPool2d(1),

            nn.Lambda(lambda x: x.flatten())
        ])
        self.trunk = nn.Sequential([
            nn.Linear(512, 256, key=keys[4]),
            nn.LayerNorm((256,), use_weight=False, use_bias=False),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(256, 256, key=keys[5]),
            nn.LayerNorm((256,), use_weight=False, use_bias=False),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(256, self.action_dim, key=keys[6]),
        ])

    def __call__(self, x: jax.Array):
        x = jnp.permute_dims(x, (2, 0, 1))  # [H, W, C] => [C, H, W]
        x = self.cnn(x)
        x = self.trunk(x)
        return x


class BoltzmannActor(eqx.Module):
    """Actor that follows a boltzmann (softmax) policy based on the Q values"""
    model: QNetwork

    def __call__(self, x: jax.Array, temperature: jax.Array, key: jax.Array):
        logits = self.model(x)
        # normalize logits, automatic temperature scaling
        logits = logits / (1e-7 + jnp.std(logits))
        return jax.random.categorical(key, logits / temperature)


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
    """Holds things for training"""
    model: eqx.Module
    opt: optax.GradientTransformation
    opt_state: optax.OptState
    gamma: jax.Array

    def update_model(self, grad):
        """Given the gradient, applies an update to the model and optimizer parameters"""
        updates, opt_state = self.opt.update(
            grad, self.opt_state, params=eqx.filter(self.model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(self.model, updates)
        return self.replace(model=model, opt_state=opt_state)


class RolloutState(State):
    """Holds things for the rollout"""
    env: Any
    wrapped_env: Any
    env_state: jax.Array
    actor: BoltzmannActor
    visualizer: Any
    obs: jax.Array
    temperature: jax.Array


class Transition(State):
    """RL transition"""
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array
    info: Any


def scan_env(rollout_state: RolloutState, num_steps: int, key: jax.random.PRNGKey) -> Tuple[RolloutState,]:
    """Scan over the environment. This is designed for a single env, make sure to use vmap!"""
    static, params = eqx.partition(rollout_state.actor, eqx.is_inexact_array)

    def step_fn(carry, _):
        obs, env_state, key = carry
        key, action_key, env_key = jax.random.split(key, 3)
        actor = eqx.combine(static, params)
        # gymnax will automatically reset done envs
        obs = rollout_state.visualizer.render(env_state)
        action = actor(obs, rollout_state.temperature, action_key)
        next_obs, next_env_state, reward, done, info = rollout_state.env.step(env_key, env_state, action)
        next_obs = rollout_state.visualizer.render(next_env_state)

        transition = Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, info=info)

        return (next_obs, next_env_state, key), transition

    (next_obs, next_env_state, key), transitions = jax.lax.scan(f=step_fn,
        init=(rollout_state.obs, rollout_state.env_state, key),
        xs=None, length=num_steps)
    rollout_state = rollout_state.replace(obs=next_obs, env_state=next_env_state)
    # Separate obs from train state as we only want to vmap over next_obs
    return rollout_state, transitions


def update(rollout_state, train_state, num_steps, key):
    """Performs a rollout and one update of the model"""
    num_envs = rollout_state.obs.shape[0]
    # Collect train data
    rollout_keys = jax.random.split(key, num_envs)
    rollout_state, transition = eqx.filter_vmap(
        scan_env,
        in_axes=(
            # You can vmap/jax/jit across dataclasses/state
            # Just construct a new dataclass/state with in_axes like so
            RolloutState(env=None, wrapped_env=None, env_state=0, actor=None, visualizer=None, obs=0, temperature=0),
            None, 0
        )
    )(rollout_state, num_steps, rollout_keys)
    #rollout_state = rollout_state.replace(temperature=rollout_state.temperature[0])

    # Update policy
    def loss_fn(model):
        # We differentiate with respect to the model, so it must be an argument
        vmodel = eqx.filter_vmap(eqx.filter_vmap(model))
        next_q = vmodel(transition.next_obs).max(axis=-1)
        target = jax.lax.stop_gradient(transition.reward + (1 - transition.done) * train_state.gamma * next_q)
        q_vals = jnp.take(vmodel(transition.obs), transition.action)
        loss = jnp.mean(jnp.square(q_vals - target))
        return loss, {"loss": loss, "mean_q": jnp.mean(q_vals), "return": jnp.mean(transition.reward)}

    grad, metrics = eqx.filter_grad(loss_fn, has_aux=True)(train_state.model)
    train_state = train_state.update_model(grad)
    rollout_state = rollout_state.replace(
        actor=BoltzmannActor(model=train_state.model),
    )

    return rollout_state, train_state, metrics

# Hyperparameters
num_envs = 16
train_epochs = 500
rollout_steps = 16
gamma = 0.99
temp = jnp.linspace(0.01, 1.00, num_envs)
lr = 1e-4
key = jax.random.key(0)
basic_env, env_params = popjym.make("CartPoleEasy")
env = LogWrapper(basic_env)
visualizer = popjym.make_render("CartPoleRender")

# Setup
env_params, init_state = jax.vmap(env.reset)(jax.random.split(key, num_envs))
init_obs = jax.vmap(visualizer.render)(init_state)
model = QNetwork(action_dim=env.action_space().n, key=key)
actor = BoltzmannActor(model=model)
opt = optax.adamw(lr)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

rstate = RolloutState(
    env = env,
    env_state = init_state,
    wrapped_env = LogWrapper(env),
    actor = actor,
    visualizer = visualizer,
    obs = init_obs,
    temperature = temp,
)
tstate = TrainState(
    model = model,
    opt = opt,
    opt_state = opt_state,
    gamma = jnp.array(gamma)
)
config = {
    "NUM_ENVS": num_envs,
    "TRAIN_EPOCHS": train_epochs,
    "ROLLOUT_STEPS": rollout_steps,
    "GAMMA": gamma,
    "TEMPERATURE": temp[0],
    "LEARNING_RATE": lr,
    "PROJECT": "popjym-acrade-test2",
    "WANDB_MODE": "online",
    "ENV_NAME": "CartPoleEasy",
    
}
for epoch in range(train_epochs):
    wandb.init(
        project=config["PROJECT"],
        tags=["PQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'pqn_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    key, subkey = jax.random.split(key)
    rstate, tstate, metrics = eqx.filter_jit(update)(rstate, tstate, num_steps=rollout_steps, key=subkey)
    metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    print(f"Epoch {epoch}: {metrics_str}")

    def callback(metrics):
        if epoch % 100 == 0:
            wandb.log(metrics)

    jax.debug.callback(callback, metrics)
