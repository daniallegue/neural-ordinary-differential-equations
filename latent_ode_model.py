import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, lax
import optax
import diffrax
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any

lr = 0.001
optimizer = optax.adam(lr)

def init_linear_params(key : random.PRNGKey,
                       in_dim : int, out_dim : int) -> Dict[str, jnp.ndarray]:
    """
    Init linear layers params

    :param key: PRNG Key
    :param in_dim: Input dim
    :param out_dim: Output dim
    :return: Param dict
    """
    W = random.normal(key, (in_dim, out_dim)) * 0.1
    b = jnp.zeros(out_dim)

    return {"W" : W, "b" : b}

def linear_forward(params : Dict[str, jnp.ndarray], x : jnp.ndarray) -> jnp.ndarray:
    """
    Applies forward layer

    :param params: Param dict
    :param x: Current state
    :return: Applied layer value
    """

    return jnp.dot(x, params["W"]) + params["b"]


def init_gru_params(
        key : random.PRNGKey,
        input_dim : int,
        hidden_dim : int
) -> Dict[str, jnp.ndarray]:
    """
    Init GRU params

    :param key: PRNG Key
    :param input_dim: Input dimension
    :param hidden_dim: Hidden Dimension
    :return: Param dict
    """

    keys = random.split(key, 6)

    params = {
        "W_R" : random.normal(keys[0], (input_dim, hidden_dim)) * 0.1,
        "U_R" : random.normal(keys[1], (hidden_dim, hidden_dim)) * 0.1,
        "b_R" : jnp.zeros(hidden_dim),
        "W_Z" : random.normal(keys[2], (input_dim, hidden_dim)) * 0.1,
        "U_Z" : random.normal(keys[3], (hidden_dim, hidden_dim)) * 0.1,
        "b_Z" : jnp.zeros(hidden_dim),
        "W_H" : random.normal(keys[4], (input_dim, hidden_dim)) * 0.1,
        "U_H" : random.normal(keys[5], (hidden_dim, hidden_dim)) * 0.1,
        "b_H" : jnp.zeros(hidden_dim)
    }

    return params

def gru_cell(
        params : Dict[str, jnp.ndarray],
        h : jnp.ndarray,
        x : jnp.ndarray
) -> jnp.ndarray:
    """
    Applies a GRU cell

    :param params: Param dict
    :param h: Hidden state
    :param x: X
    :return: New hidden state
    """

    r = jax.nn.sigmoid(jnp.dot(x, params["W_R"]) + jnp.dot(h, params["U_R"]) + params["b_R"])
    z = jax.nn.sigmoid(jnp.dot(x, params["W_Z"]) + jnp.dot(h, params["U_Z"]) + params["b_Z"])

    h_tilde = jnp.tanh(jnp.dot(x, params["W_H"]) + jnp.dot(r * h, params["U_H"]) + params["b_H"])

    h_new = (1 - z) * h + z * h_tilde

    return h_new

def gru_encoder(gru_params : Dict[str, jnp.ndarray], x_seq : jnp.ndarray) -> jnp.ndarray:
    """
    Encodes X-series

    :param gru_params: Param dict
    :param x_seq: (T, input_dim)
    :return: Final hidden state
    """

    h = jnp.zeros(gru_params["b_R"].shape)

    def step(h, x):
        h_new = gru_cell(gru_params, h, x)
        return h_new, h_new

    h_final, _ = lax.scan(step, h, x_seq)

    return h_final

# Encoder will map observed time series to params of the latent intial state
def init_encoder_params(key : random.PRNGKey, input_dim : int, gru_hidden_dim : int, latent_dim : int) -> Dict[str, Any]:
    """
    Init encoder params

    :param key: Param key
    :param input_dim: Input dim
    :param gru_hidden_dim: Hidden state dim
    :param latent_dim: Latent state dimension
    :return: Param dict
    """

    k1, k2, k3 = random.split(key, 3)
    gru_params = init_gru_params(k1, input_dim, gru_hidden_dim)
    linear_mean = init_linear_params(k2, gru_hidden_dim, latent_dim)
    linear_log_var = init_linear_params(k3, gru_hidden_dim, latent_dim)

    return {"gru" : gru_params, "mean" : linear_mean, "logvar" : linear_log_var}

def encode(encoder_params : Dict[str, Any], x_seq : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encodes X time series into latent space

    :param encoder_params: Param dict
    :param x_seq: Time series
    :return: Encoded function params
    """

    h_enc = gru_encoder(encoder_params["gru"], x_seq)
    z0_mean = linear_forward(encoder_params["mean"], h_enc)
    z0_logvar = linear_forward(encoder_params["logvar"], h_enc)

    return z0_mean, z0_logvar

def reparameterize(key: jax.random.PRNGKey, mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Reparametrize latent function

    :param key: PRNG Key
    :param mean: Function mean
    :param logvar: Function Log-Variance
    :return: Reparametrized value
    """
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(key, mean.shape)
    return mean + eps * std

def init_latent_dynamics_params(key : random.PRNGKey, latent_dim : int, hidden_dim : int) -> Dict[str, Any]:
    """
    Init latent state params

    :param key: PRNG Key
    :param latent_dim: Latent dimension
    :param hidden_dim: Hidden dimension
    :return: Param dict
    """

    k1, k2 = random.split(key, 2)
    linear1 = init_linear_params(k1, latent_dim, hidden_dim)
    linear2 = init_linear_params(k2, hidden_dim, latent_dim)

    return {"linear1" : linear1, "linear2" : linear2}

def latent_dynamics_func(params: Dict[str, Any], t: float, z: jnp.ndarray) -> jnp.ndarray:
    """
    Latent dynamics function (T-independent)

    :param params: Param dict
    :param t: T-value
    :param z: Current state
    :return: dz/dt
    """
    h = jnp.tanh(linear_forward(params["linear1"], z))
    dz_dt = linear_forward(params["linear2"], h)
    return dz_dt

def init_decoder_params(key : random.PRNGKey, latent_dim : int, hidden_dim : int, output_dim : int) -> Dict[str, Any]:
    """

    :param key: PRNGKey
    :param latent_dim: Latent dimension
    :param hidden_dim: Hidden dimension
    :param output_dim: Output dimension
    :return: Param dict
    """

    k1, k2 = random.split(key, 2)
    linear1 = init_linear_params(k1, latent_dim, hidden_dim)
    linear2 = init_linear_params(k2, hidden_dim, output_dim)

    return {"linear1" : linear1, "linear2" : linear2}

def decode(decoder_params : Dict[str, Any], z : jnp.ndarray) -> jnp.ndarray:
    """
    Decode function

    :param decoder_params: Param dict
    :param z: Current state
    :return: Decoded value
    """

    h = jnp.tanh(linear_forward(decoder_params["linear1"], z))
    x_recon = linear_forward(decoder_params["linear2"], h)

    return x_recon

def latent_ode_loss(
        encoder_params : Dict[str, Any],
        latent_dynamics_params : Dict[str, Any],
        decoder_params : Dict[str, Any],
        key : random.PRNGKey,
        x_seq : jnp.ndarray,
        t_seq : jnp.ndarray
) -> jnp.ndarray:
    """

    Computes ODE loss

    :param x_seq: Observed t-series (T, obs_dim)
    :param t_seq: time points, shape (T, ,)
    """

    z0_mean, z0_logvar = encode(encoder_params, x_seq)
    z0 = reparameterize(key, z0_mean, z0_logvar)

    t0 = t_seq[0]
    t1 = t_seq[-1]

    term = diffrax.ODETerm(lambda t, y, args : latent_dynamics_func(args, t, y))
    solver = diffrax.Dopri5()
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0 = t0,
        t1 = t1,
        dt0 = 0.1,
        y0 = z0,
        args = latent_dynamics_params,
        adjoint = adjoint,
        saveat = diffrax.SaveAt(ts = t_seq)
    )

    z_t = sol.ys

    # Decode latent trajectory
    x_recon = vmap(lambda z : decode(decoder_params, z))(z_t)
    recon_loss = jnp.mean((x_recon - x_seq) ** 2)

    # Finds divergence between standard normal prior and computed distribution
    kl_div = -0.5 * jnp.sum(1 + z0_logvar -z0_mean ** 2 - jnp.exp(z0_logvar))
    kl_div /= x_seq.shape[0]

    return recon_loss + kl_div

class LatentODEModel:
    def __init__(self,
                 input_dim: int,
                 rnn_hidden: int,
                 latent_dim: int,
                 dynamics_hidden: int,
                 decoder_hidden: int,
                 timesteps: int,
                 lr: float = 0.001,
                 key: jax.random.PRNGKey = random.PRNGKey(0)):
        self.input_dim = input_dim
        self.rnn_hidden = rnn_hidden
        self.latent_dim = latent_dim
        self.dynamics_hidden = dynamics_hidden
        self.decoder_hidden = decoder_hidden
        self.timesteps = timesteps
        self.t_seq = jnp.linspace(0.0, 1.0, timesteps)
        self.lr = lr
        self.key = key

        # Initialize model parameters.
        self.key, subkey = random.split(self.key)
        self.encoder_params = init_encoder_params(subkey, input_dim, rnn_hidden, latent_dim)
        self.key, subkey = random.split(self.key)
        self.latent_dynamics_params = init_latent_dynamics_params(subkey, latent_dim, dynamics_hidden)
        self.key, subkey = random.split(self.key)
        self.decoder_params = init_decoder_params(subkey, latent_dim, decoder_hidden, input_dim)

        # Setup optimizer.
        self.optimizer = optax.adam(lr)
        params_tuple = (self.encoder_params, self.latent_dynamics_params, self.decoder_params)
        self.opt_state = self.optimizer.init(params_tuple)

    @partial(jax.jit, static_argnums=0)
    def _train_step(self,
                    encoder_params: Dict[str, Any],
                    latent_dynamics_params: Dict[str, Any],
                    decoder_params: Dict[str, Any],
                    opt_state: optax.OptState,
                    key: jax.random.PRNGKey,
                    x_batch: jnp.ndarray,  # shape (batch_size, T, obs_dim)
                    t_seq: jnp.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], optax.OptState, jnp.ndarray]:
        params_tuple = (encoder_params, latent_dynamics_params, decoder_params)
        def loss_fn(params_tuple: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]) -> jnp.ndarray:
            enc_params, dyn_params, dec_params = params_tuple
            batch_keys = random.split(key, x_batch.shape[0])
            losses = vmap(
                lambda x_seq, k: latent_ode_loss(enc_params, dyn_params, dec_params, k, x_seq, t_seq)
            )(x_batch, batch_keys)
            return jnp.mean(losses)
        loss, grads = jax.value_and_grad(loss_fn)(params_tuple)
        updates, opt_state = self.optimizer.update(grads, opt_state, params_tuple)
        new_params = optax.apply_updates(params_tuple, updates)
        return new_params[0], new_params[1], new_params[2], opt_state, loss

    def train(self, x_data: jnp.ndarray, num_epochs: int, batch_size: int) -> Tuple[List[float], List[float]]:
        """
        Trains the latent ODE model on x_data.
        x_data: shape (num_samples, T, obs_dim)
        """
        num_samples = x_data.shape[0]
        num_batches = num_samples // batch_size
        loss_history: List[float] = []
        time_history: List[float] = []
        for epoch in range(num_epochs):
            start_time = time.time()
            perm_key = random.PRNGKey(epoch)
            perm = random.permutation(perm_key, num_samples)
            x_data_shuffled = x_data[perm]
            epoch_loss = 0.0
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_data_shuffled[start:end]
                self.encoder_params, self.latent_dynamics_params, self.decoder_params, self.opt_state, loss = self._train_step(
                    self.encoder_params, self.latent_dynamics_params, self.decoder_params,
                    self.opt_state, self.key, x_batch, self.t_seq
                )
                epoch_loss += loss
            epoch_loss /= num_batches
            end_time = time.time()
            epoch_time = end_time - start_time
            loss_history.append(float(epoch_loss))
            time_history.append(epoch_time)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f} sec")
        return loss_history, time_history

    def predict(self, x_seq: jnp.ndarray, pred_timesteps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts trajectory for a given input sequence.
        x_seq: shape (T, input_dim)
        pred_timesteps: number of timesteps for prediction (can be > T for extrapolation)
        Returns: (x_pred, t_pred)
        """
        self.key, subkey = random.split(self.key)
        z0_mean, z0_logvar = encode(self.encoder_params, x_seq)
        z0 = reparameterize(subkey, z0_mean, z0_logvar)
        t_pred = jnp.linspace(0.0, 1.0, pred_timesteps)
        t0 = t_pred[0]
        term = diffrax.ODETerm(lambda t, y, args: latent_dynamics_func(args, t, y))
        solver = diffrax.Dopri5()
        adjoint = diffrax.RecursiveCheckpointAdjoint()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(t0),
            t1=float(t_pred[-1]),
            dt0=0.1,
            y0=z0,
            args=self.latent_dynamics_params,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(ts=t_pred)
        )
        z_t = sol.ys
        x_pred = vmap(lambda z: decode(self.decoder_params, z))(z_t)
        return x_pred, t_pred




