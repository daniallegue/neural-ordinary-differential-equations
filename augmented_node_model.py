# augmented_node_model.py
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import random
import diffrax

def init_linear_params(key : jax.random.PRNGKey, in_dim : int, out_dim : int) -> Dict:
    """

    Initialize weights and biases for linear layer
    :param key: PRNG Key
    :param in_dim: Input dimension
    :param out_dim: Output dimension
    :return: Param dict
    """

    W = random.normal(key, (in_dim, out_dim)) * 0.1
    b = jnp.zeros(out_dim)

    return {'W' : W, 'b' : b}

def init_conv_params(key : jax.random.PRNGKey, filter_shape : Tuple) -> Dict:
    """

    :param key: PRNG Key
    :param filter_shape: Tuple (kernel_h, kernel_w, in_channels, out_channels)
    :return: Param dict
    """

    W = random.normal(key, filter_shape) * 0.1
    b = jnp.zeros(filter_shape[-1]) # i.e. out_channels

    return {'W' : W, 'b' : b}

def linear_forward(params : dict, x : jnp.ndarray) -> jnp.ndarray:
    """
    Applies linear layer forward

    :param params: Param dict
    :param x: X
    :return: value
    """

    return jnp.dot(x, params["W"]) + params["b"]

def conv_forward(params : Dict, x : jnp.ndarray, stride : Tuple = (1, 1), padding : str = "SAME") -> jnp.ndarray:
    """
    Apply a convolutional layer.

    :param params: Dictionary with "w" and "b".
    :param x: Input image of shape (H, W, in_channels).
    :param stride: Stride for the convolution.
    :param padding: Padding, e.g. "SAME" or "VALID".
    :return: Output of the convolution with shape (H, W, out_channels).
    """

    x = jnp.expand_dims(x, axis=0)
    y = jax.lax.conv_general_dilated(
        x,
        params['W'],
        window_strides=stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )

    y = y + params['b']
    return jnp.squeeze(y, axis=0) # Remove batch dimension


def init_augmented_dynamics_params(key: jax.random.PRNGKey, input_dim: int, hidden_dim: int, output_dim: int) -> dict:
    """
    Initialize parameters for the dynamics function.
    We use a one-hidden-layer MLP with a tanh activation.

    :param input_dim: Dimension of the augmented state.
    :param hidden_dim: Number of hidden units.
    :param output_dim: Dimension of the output (typically equal to input_dim).
    """
    key1, key2 = random.split(key)
    linear1 = init_linear_params(key1, input_dim, hidden_dim)
    linear2 = init_linear_params(key2, hidden_dim, output_dim)
    return {"linear1": linear1, "linear2": linear2}


def augmented_dynamics_func(params: dict, t: float, z: jnp.ndarray) -> jnp.ndarray:
    """
    Computes dz/dt for the augmented neural ODE.

    z is the augmented state, i.e. a concatenation of the original state x and
    the augmented part a. The dynamics function is given by:

        f(t, z) = linear2( tanh( linear1(z) ) )
    """
    h = jnp.tanh(linear_forward(params["linear1"], z))
    dz_dt = linear_forward(params["linear2"], h)
    return dz_dt


def conv_augmented_dynamics_func(params : Dict, t : float, z : jnp.ndarray) -> jnp.ndarray:
    """

    Convolutional dynamics function for Conv ANODE.

    z is the augmented image tensor of shape (H, W, total_channels).
    The function applies two convolutional layers with a ReLU activation in between.
    """

    h = jax.nn.relu(conv_forward(params["conv1"], z, stride = (1,1), padding="SAME"))
    dz_dt = conv_forward(params["conv2"], h, stride=(1,1), padding="SAME")

    return dz_dt
class AugmentedNeuralODEModel:

    def __init__(self, orig_dim : int, aug_dim : int, hidden_dim : int, lr : float = 0.001, key : jax.random.PRNGKey = random.PRNGKey(0), solver = None):
        """
        Augmented Neural ODE Model.

        :param orig_dim: Dimension of the original state.
        :param aug_dim: Number of augmentation dimensions.
        :param hidden_dim: Number of hidden units in the dynamics MLP.
        :param lr: Learning rate (for later use in training).
        :param key: JAX random key.
        :param solver: diffrax ODE solver to use; defaults to diffrax.Dopri5.
        """
        self.orig_dim = orig_dim
        self.aug_dim = aug_dim
        self.augmented_dim = orig_dim + aug_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.key = key

        self.key, subkey = random.split(self.key)
        self.dynamics_params = init_augmented_dynamics_params(subkey, self.augmented_dim, hidden_dim, self.augmented_dim)

        self.solver = solver if solver is not None else diffrax.Dopri5()

    def integrate(self, x0 : jnp.ndarray, t0 : float, t1 : float, t_eval : jnp.ndarray) -> jnp.ndarray:
        """

        :param x0: Initial state (shape: (orig_dim,))
        :param t0: Initial time.
        :param t1: Final time.
        :param t_eval: 1D array of time points at which to save the solution.
        :return: The integrated original state trajectories (shape: (len(t_eval), orig_dim)).
        """

        zeros_aug = jnp.zeros(self.aug_dim)
        z0 = jnp.concatenate([x0, zeros_aug], axis = 0)

        term = diffrax.ODETerm(lambda t, z, args : augmented_dynamics_func(args, t ,z))
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0 = t0,
            t1 = t1,
            dt0 = 0.1,
            y0 = z0,
            args = self.dynamics_params,
            saveat = diffrax.SaveAt(ts = t_eval)
        )

        z_t = sol.ys
        x_t = z_t[:, :self.orig_dim] # Remove augmented dimensions
        return x_t

    @partial(jax.jit, static_args = 0)
    def train_step(self,
                   x: jnp.ndarray,
                   t0: float = 0.0,
                   t1: float = 1.0,
                   t_eval: jnp.ndarray = jnp.array([0.0, 1.0]),
                   optimizer=None,
                   opt_state=None) -> Tuple[jnp.ndarray, optax.OptState]:
        """
        Perform one training step on a single sample x.

        :param x: Input state (shape: (orig_dim,))
        :param t0: Initial time.
        :param t1: Final time.
        :param t_eval: 1D array of time points to save the solution.
        :param optimizer: An optax optimizer.
        :param opt_state: The current optimizer state.
        :return: Tuple (loss, new optimizer state)
        """

        def loss_fn(params):
            zeros_aug = jnp.zeros(self.aug_dim)
            z0 = jnp.concatenate([x, zeros_aug], axis=0)
            term = diffrax.ODETerm(lambda t, z, args: augmented_dynamics_func(args, t, z))
            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                t0=t0,
                t1=t1,
                dt0=0.1,
                y0=z0,
                args=params,
                saveat=diffrax.SaveAt(ts=t_eval)
            )
            z_t = sol.ys
            x_pred = z_t[:, :self.orig_dim]
            return jnp.mean((x_pred - x) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(self.dynamics_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        self.dynamics_params = optax.apply_updates(self.dynamics_params, updates)
        return loss, new_opt_state


class ConvAugmentedNODE:

    def __init__(self, image_shape : Tuple, aug_channels : int, hidden_channels : int, kernel_size : int = 3, lr : float = 0.001, key : jax.random.PRNGKey = random.PRNGKey(0), solver = None):
        """

        Convolutional Augmented Neural ODE Model.

        :param image_shape: Tuple (H, W, C) for the original image.
        :param aug_channels: Number of augmentation channels to be concatenated.
        :param hidden_channels: Number of channels in the hidden convolution layer.
        :param kernel_size: Kernel size for the convolution layers.
        :param lr: Learning rate (for later use).
        :param key: JAX random key.
        :param solver: diffrax ODE solver to use; defaults to diffrax.Dopri5.
        """

        self.image_shape = image_shape # (H, W , C)
        self.orig_channels = image_shape[-1]
        self.aug_channels = aug_channels
        self.total_channels = self.orig_channels + aug_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.lr = lr
        self.key = key

        key, subkey1, subkey2 = random.split(self.key, 3)
        conv1_shape = (kernel_size, kernel_size, self.total_channels, hidden_channels)
        conv2_shape = (kernel_size, kernel_size, hidden_channels, self.total_channels)
        self.dynamics_params = {
            "conv1": init_conv_params(subkey1, conv1_shape),
            "conv2": init_conv_params(subkey2, conv2_shape)
        }

        self.solver = solver if solver is not None else diffrax.Dopri5()

    def integrate(self, x0 : jnp.ndarray, t0 : float, t1 : float, t_eval : jnp.ndarray) -> jnp.ndarray:
        """
        Integrate the Conv ANODE from t0 to t1 given an initial image x0.

        The original image x0 (shape: (H, W, C)) is augmented by concatenating zeros along the channel dimension.

        :param x0: Initial image (shape: (H, W, C)).
        :param t0: Initial time.
        :param t1: Final time.
        :param t_eval: 1D array of time points at which to save the solution.
        :return: Integrated original image trajectories (shape: (len(t_eval), H, W, C)).
        """

        zeros_aug = jnp.zeros((*x0.shape[:2], self.aug_channels))
        z0 = jnp.concatenate([x0, zeros_aug], axis = -1)

        term = diffrax.ODETerm(lambda t, z, args : conv_augmented_dynamics_func(args, t, z))
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0 = t0,
            t1 = t1,
            dt0 = 0.1,
            y0 = z0,
            args = self.dynamics_params,
            saveat = diffrax.SaveAt(ts = t_eval)
        )

        z_t = sol.ys
        x_t = z_t[..., :self.orig_channels]
        return x_t

    @partial(jax.jit, static_args = 0)
    def train_step(self,
                   x: jnp.ndarray,
                   t0: float = 0.0,
                   t1: float = 1.0,
                   t_eval: jnp.ndarray = jnp.array([0.0, 1.0]),
                   optimizer=None,
                   opt_state=None) -> Tuple[jnp.ndarray, optax.OptState]:
        """
        Perform one training step on a single image x.

        :param x: Input image (shape: (H, W, C)).
        :param t0: Initial time.
        :param t1: Final time.
        :param t_eval: 1D array of time points to save the solution.
        :param optimizer: An optax optimizer.
        :param opt_state: The current optimizer state.
        :return: Tuple (loss, new optimizer state)
        """

        def loss_fn(params):
            zeros_aug = jnp.zeros((*x.shape[:2], self.aug_channels))
            z0 = jnp.concatenate([x, zeros_aug], axis=-1)
            term = diffrax.ODETerm(lambda t, z, args: conv_augmented_dynamics_func(args, t, z))
            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                t0=t0,
                t1=t1,
                dt0=0.1,
                y0=z0,
                args=params,
                saveat=diffrax.SaveAt(ts=t_eval)
            )
            z_t = sol.ys
            x_pred = z_t[..., :self.orig_channels]
            return jnp.mean((x_pred - x) ** 2) # Return loss

        loss, grads = jax.value_and_grad(loss_fn)(self.dynamics_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        self.dynamics_params = optax.apply_updates(self.dynamics_params, updates)
        return loss, new_opt_state