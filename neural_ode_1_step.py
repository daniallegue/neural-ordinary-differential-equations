from datetime import time

import jax
import jax.numpy as jnp
from jax import random, grad, jit
from typing import Dict, Tuple
import optax
import diffrax
from matplotlib import pyplot as plt

# Define an optimizer
lr = 0.001
optimizer = optax.adam(lr)

def initialize_params(
        key: jax.random.PRNGKey,
        input_dim : int,
        hidden_dim : int,
        output_dim : int
) -> Dict[str, jnp.ndarray]:
    """
    Initializes the params of the network randomly
    :param key: PRNG Key
    :param input_dim: Input dimension
    :param hidden_dim: Hidden Layer Dimension
    :param output_dim: Output Dimension
    :return: Param Dict
    """

    keys = random.split(key, 2)
    params: Dict[str, jnp.ndarray] = {
        'W1' : random.normal(keys[0], (input_dim, hidden_dim)),
        'b1' : jnp.zeros(hidden_dim),
        'W2' : random.normal(keys[1], (hidden_dim, output_dim)),
        'b2' : jnp.zeros(output_dim)
    }

    return params

def dynamics_func(
        params : Dict[str, jnp.ndarray],
        t : float,
        z : jnp.ndarray
) -> jnp.ndarray:
    """
    Implements the dynamics function
    :param params: Param Dict
    :param t: time-dependent dynamics param
    :param z: state
    :return: dz/dt
    """

    # Introduce tanh for non-linearity
    hidden = jnp.tanh(jnp.dot(z, params['W1']) + params['b1'])

    # Linear transformation
    dz_dt = jnp.dot(hidden, params['W2']) + params['b2']
    return dz_dt

def loss_func(
    params : Dict[str, jnp.ndarray],
    z0 : jnp.ndarray,
    t0 : jnp.ndarray,
    t1 : jnp.ndarray,
    target : jnp.ndarray
) -> jnp.ndarray:

    # Wrap the dynamics function into an ODE term
    term = diffrax.ODETerm(lambda t, y, args: dynamics_func(args, t, y))
    solver = diffrax.Dopri5()

    # Utilise adjoint method
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0 = t0,
        t1 = t1,
        dt0 = 0.1,
        y0 = z0,
        args = params,
        adjoint = adjoint,
        saveat = diffrax.SaveAt(t1 = True) # We only need to solve at the last t (t1)
    )

    zT = sol.ys # Final state at t1

    loss = jnp.mean((zT - target) ** 2)
    return loss

@jax.jit
def train_step(
    params : Dict[str, jnp.ndarray],
    opt_state : optax.OptState,
    z0 : jnp.ndarray,
    t0 : float,
    t1 : float,
    target : jnp.ndarray
) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
    """
    Single training state

    :param params: Current model params
    :param opt_state: Current optimizer state
    :param z0: Initial ODE state
    :param t0: Start time for ODE
    :param t1: End time for ODE
    :param target: Target state at t1
    :return: [Updated params, Updated optimizer, Scalar loss for param]
    """

    loss, grads = jax.value_and_grad(loss_func)(params, z0, t0, t1, target)

    updates, opt_state = optimizer.update(grads, opt_state, params)

    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss
