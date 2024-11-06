"""JAX compatible version of CartPole-v1 OpenAI gym environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import functools
from jax.tree_util import register_pytree_node_class
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    movement: int
    score: int
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500  # v0 had only 200 steps!


# @register_pytree_node_class
class CartPole(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of CartPole-v1 OpenAI gym environment.


    Source: github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (3, 256, 256)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        # force = params.force_mag * action - params.force_mag * (1 - action)
        # force = params.force_mag * (action - 3.5) * 2
        force = lax.cond(action == 2,
                         lambda _: -params.force_mag,
                         lambda _: lax.cond(action == 3,
                                            lambda _: params.force_mag,
                                            lambda _: 0.0,
                                            operand=None),
                         operand=None)

        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Only default Euler integration option available here!
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal

        def map_to_movement(value):
            value = lax.cond(value > 0,
                             lambda _: 4 * value + 1,
                             lambda _: 4 * value - 1,
                             operand=None)
            return value

        movement = state.movement + map_to_movement(state.x).astype(int)
        # Update state dict and evaluate termination conditions
        done = self.is_terminal(state, params)
        new_score = state.score + lax.cond(reward > 0, lambda _: 1, lambda _: 0, operand=None)
        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            movement=movement,
            time=state.time + 1,
            score=new_score,
        )
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def get_partial(self) -> bool:
        return self.partial_observable

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            movement=0,
            score=0,
            time=0,
        )
        init_obs = jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])
        return init_obs, state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if self.partial_observable:
            high = jnp.array(
                [
                    jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                ]
            )
            return spaces.Box(-high, high, shape=(2,), dtype=jnp.float32)
        else:
            high = jnp.array(
                [
                    params.x_threshold * 2,
                    jnp.finfo(jnp.float32).max,
                    params.theta_threshold_radians * 2,
                    jnp.finfo(jnp.float32).max,
                ]
            )
            return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
    #
    # def tree_flatten(self):
    #     children = ()
    #     aux_data = (self.obs_shape, )
    #     return children, aux_data
    #
    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*children)

