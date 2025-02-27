import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from typing import Dict, List, Tuple
import optax
import diffrax
import time
from matplotlib import pyplot as plt

# Define optimizer for later use
learning_rate = 0.001
optimizer = optax.adam(learning_rate)
def init_params(
        key : jax.random.PRNGKey,
        input_dim : int,
        hidden_dims : List[int],
        output_dims : int
) -> Dict[str, jnp.ndarray]:
    """
    Initialize params for MLP with arbitrary number of layers

    :param key: PRNG
    :param input_dim: Input dim
    :param hidden_dims: Hidden dims
    :param output_dims: Output dims
    :return: Params Dict
    """

    num_layers = len(hidden_dims) + 1
    keys = random.split(key, num_layers)

    params :  Dict[str, jnp.ndarray] = {}

    dims = [input_dim] + hidden_dims + [output_dims]

    for i in range(num_layers):
        params[f"W{i+1}"] = random.normal(keys[i], (dims[i], dims[i+1]))
        params[f"b{i+1}"] = jnp.zeros(dims[i+1])

    return params

def dynamics_fn(
        params : Dict[str, jnp.ndarray],
        t : float,
        z : jnp.ndarray
) -> jnp.ndarray:
    """
    Computes dz/dt

    :param params: Param dict
    :param t: Current t
    :param z: Current state z
    :return: dz/dt
    """

    num_layers = len(params) // 2
    x = z

    for i in range(1, num_layers):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        x = jnp.tanh(jnp.dot(x, W) + b) # Introduce non-linearity

    W_final = params[f"W{num_layers}"]
    b_final = params[f"b{num_layers}"]

    return jnp.dot(x, W_final) + b_final

def integrate_single(
        params : Dict[str, jnp.ndarray],
        z0 : jnp.ndarray,
        t0 : float,
        t1 : float
) -> jnp.ndarray:
    """
    Integrates ODE from t0 to t1

    :param params: Current param dict
    :param z0: Initial state
    :param t0: start t
    :param t1: end t
    :return: final state at t1
    """

    term = diffrax.ODETerm(lambda t, y, args : dynamics_fn(params, t, y))
    solver = diffrax.Dopri5()

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
        saveat = diffrax.SaveAt(t1 = True)
    )

    return sol.ys

def loss_fn_single_step(
        params : Dict[str, jnp.ndarray],
        z0 : jnp.ndarray,
        t0 : float,
        t1 : float,
        target : jnp.ndarray
) -> jnp.ndarray:
    """
    Mean-Square Loss

    :param params: Current params
    :param z0: Initial state
    :param t0: Initial time
    :param t1: Final time
    :param target: Target value
    :return: MS Loss
    """

    z_t = integrate_single(params, z0, t0, t1)
    return jnp.mean((z_t - target) ** 2)

# Vectorize loss_fn over batches
loss_fn_batch = lambda params, z0_batch, t0, t1, target_batch : jnp.mean(
    vmap(loss_fn_single_step, in_axes=(None, 0, None, None, 0))(params, z0_batch, t0, t1, target_batch)
)

@jit
def train_step(
        params : Dict[str, jnp.ndarray],
        opt_state : optax.OptState,
        z0_batch : jnp.ndarray,
        t0 : float,
        t1 : float,
        target_batch : jnp.ndarray
) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
    """
    Performs one training step

    :param params: Param dict
    :param opt_state: Optimizer state
    :param z0_batch: Batch of intial state
    :param t0: Initial time
    :param t1: Final time
    :param target_batch: Batch of target state
    :return: Updated Params, Optimizer state, Loss
    """

    loss, grads = jax.value_and_grad(loss_fn_batch)(params, z0_batch, t0, t1, target_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss


def generate_synthetic_dataset(
        key : random.PRNGKey,
        num_samples : int,
        input_dim : int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates a synthetic dataset for which dz/dt = -z.
    For an initial state z0, z(1) = exp(-1) * z0

    :param key: RNG Key
    :param num_samples: Number of samples to generate
    :param input_dim: Sample Dimension
    :return: Initial state, Target
    """

    z0_data = random.normal(key, (num_samples, input_dim))

    # True Solution
    target_data = jnp.exp(-1) * z0_data

    return z0_data, target_data

def train_model(
        params : Dict[str, jnp.ndarray],
        opt_state : optax.OptState,
        z0_data : jnp.ndarray,
        target_data : jnp.ndarray,
        t0 : float,
        t1 : float,
        num_epochs : int,
        batch_size : int
) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, list, list]:
    """
    Train Method

    :param params: Param dict
    :param opt_state: Optimizer state
    :param z0_data: Inital data
    :param target_data: Target data
    :param t0: Initial t
    :param t1: Final t
    :param num_epochs: Number of epochs
    :param batch_size: Batch size
    :return: Updated params, Optimizer state
    """

    num_samples = z0_data.shape[0]
    num_batches = num_samples // batch_size

    loss_history = []
    time_history = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        perm_key = random.PRNGKey(epoch)
        perm = random.permutation(perm_key, num_samples)

        # Permute data
        z0_data_shuffled = z0_data[perm]
        target_data_shuffled = target_data[perm]

        epoch_loss = 0.0
        for i in range(num_epochs):
            start = i * batch_size
            end = start + batch_size

            z0_batch = z0_data_shuffled[start:end]
            target_batch = target_data_shuffled[start:end]
            params, opt_state, loss = train_step(params, opt_state, z0_batch, t0, t1, target_batch)
            epoch_loss += loss
        epoch_loss /= num_batches
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        loss_history.append(epoch_loss)
        time_history.append(epoch_time)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f} sec")

    return params, opt_state, loss_history, time_history

key = random.PRNGKey(42)
input_dim = 4
hidden_dim = 16
output_dim = 4

# Initialize model parameters.
params = init_params(key, input_dim, [16], output_dim)

num_epochs = 40
batch_size = 32

opt_state = optimizer.init(params)

num_samples = 10000
z0_data, target_data = generate_synthetic_dataset(key, num_samples, input_dim)

t0 = 0.0
t1 = 1.0


params, opt_state, loss_history, time_history = train_model(params, opt_state,
    z0_data, target_data, t0, t1, num_epochs, batch_size)


# Plot training loss and epoch times.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(time_history, marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Time (sec)")
plt.title("Epoch Training Time")

plt.tight_layout()
plt.show()

