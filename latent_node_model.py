import time
from functools import partial
import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import optax
import diffrax
from typing import Dict, Tuple, List, Any

def init_linear_params(key: jax.random.PRNGKey,
                       in_dim: int,
                       out_dim: int) -> Dict[str, jnp.ndarray]:
    """Initialize weights and biases for a linear layer."""
    W = random.normal(key, (in_dim, out_dim)) * 0.1
    b = jnp.zeros(out_dim)
    return {"W": W, "b": b}


def conv_node_init_conv_params(key: jax.random.PRNGKey, filter_shape: tuple) -> dict:
    """
    Initialize parameters for a convolutional layer.

    :param filter_shape: Tuple (kernel_h, kernel_w, in_channels, out_channels).
    :return: Dictionary with weights and biases.
    """
    w = random.normal(key, filter_shape) * 0.1
    b = jnp.zeros(filter_shape[-1])
    return {"w": w, "b": b}

def linear_forward(params: Dict[str, jnp.ndarray],
                   x: jnp.ndarray) -> jnp.ndarray:
    """Apply a linear layer: out = xW + b."""
    return jnp.dot(x, params["W"]) + params["b"]

def conv_node_forward(params: dict, x: jnp.ndarray, stride: tuple = (1, 1), padding: str = "SAME") -> jnp.ndarray:
    """
    Apply a convolutional layer.

    :param params: Dictionary with "w" and "b".
    :param x: Input image tensor of shape (H, W, in_channels).
    :param stride: Stride for the convolution.
    :param padding: Padding, e.g. "SAME" or "VALID".
    :return: Output tensor of shape (H, W, out_channels).
    """
    # Add a batch dimension.
    x = jnp.expand_dims(x, axis=0)  # (1, H, W, in_channels)
    y = jax.lax.conv_general_dilated(
        x,
        params["w"],
        window_strides=stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    y = y + params["b"]
    return jnp.squeeze(y, axis=0)

def init_gru_params(key: jax.random.PRNGKey,
                    input_dim: int,
                    hidden_dim: int) -> Dict[str, jnp.ndarray]:
    """Initialize parameters for a single GRU layer."""
    keys = random.split(key, 6)
    return {
        "W_R": random.normal(keys[0], (input_dim, hidden_dim)) * 0.1,
        "U_R": random.normal(keys[1], (hidden_dim, hidden_dim)) * 0.1,
        "b_R": jnp.zeros(hidden_dim),
        "W_Z": random.normal(keys[2], (input_dim, hidden_dim)) * 0.1,
        "U_Z": random.normal(keys[3], (hidden_dim, hidden_dim)) * 0.1,
        "b_Z": jnp.zeros(hidden_dim),
        "W_H": random.normal(keys[4], (input_dim, hidden_dim)) * 0.1,
        "U_H": random.normal(keys[5], (hidden_dim, hidden_dim)) * 0.1,
        "b_H": jnp.zeros(hidden_dim)
    }

def gru_cell(params: Dict[str, jnp.ndarray],
             h: jnp.ndarray,
             x: jnp.ndarray) -> jnp.ndarray:
    """One step of GRU update: h_new = GRUCell(h, x)."""
    r = jax.nn.sigmoid(jnp.dot(x, params["W_R"]) + jnp.dot(h, params["U_R"]) + params["b_R"])
    z = jax.nn.sigmoid(jnp.dot(x, params["W_Z"]) + jnp.dot(h, params["U_Z"]) + params["b_Z"])
    h_tilde = jnp.tanh(jnp.dot(x, params["W_H"]) + jnp.dot(r * h, params["U_H"]) + params["b_H"])
    h_new = (1 - z) * h + z * h_tilde
    return h_new

def gru_encoder(gru_params: Dict[str, jnp.ndarray],
                x_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Encode a time-series x_seq (T, input_dim) into a final hidden state (hidden_dim).
    """
    h0 = jnp.zeros(gru_params["b_R"].shape)
    def step(h, x):
        h_new = gru_cell(gru_params, h, x)
        return h_new, h_new
    h_final, _ = lax.scan(step, h0, x_seq)
    return h_final

def init_biencoder_params(key: jax.random.PRNGKey,
                          input_dim: int,
                          rnn_hidden_dim: int,
                          latent_dim: int) -> Dict[str, Any]:
    """Initialize GRU-based encoder that outputs mean and log-variance for z0."""
    k1, k2, k3, k4 = random.split(key, 4)
    gru_params_fwd = init_gru_params(k1, input_dim, rnn_hidden_dim)
    gru_params_bwd = init_gru_params(k2, input_dim, rnn_hidden_dim)
    linear_mean = init_linear_params(k3, 2 * rnn_hidden_dim, latent_dim)
    linear_logvar = init_linear_params(k4, 2 * rnn_hidden_dim, latent_dim)
    return {"gru_fwd": gru_params_fwd, "gru_bwd": gru_params_bwd, "mean": linear_mean, "logvar": linear_logvar}

def bi_gru_encoder(encoder_params : Dict[str, Any],
                   x_seq : jnp.ndarray) -> jnp.ndarray:
    """
    Run both forward and backward GRU encoders on x_seq and concatenate their final hidden states.
    """
    h_fwd = gru_encoder(encoder_params["gru_fwd"], x_seq)
    h_bwd = gru_encoder(encoder_params["gru_bwd"], x_seq[::-1])
    return jnp.concatenate([h_fwd, h_bwd], axis=-1)

def encode(encoder_params: Dict[str, Any],
           x_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Encode x_seq into z0_mean, z0_logvar."""
    h_enc = bi_gru_encoder(encoder_params, x_seq)
    z0_mean = linear_forward(encoder_params["mean"], h_enc)
    z0_logvar = linear_forward(encoder_params["logvar"], h_enc)
    return z0_mean, z0_logvar

def reparameterize(key: jax.random.PRNGKey,
                   mean: jnp.ndarray,
                   logvar: jnp.ndarray) -> jnp.ndarray:
    """Re-parameterization trick: z0 = mean + eps * std."""
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(key, mean.shape)
    return mean + eps * std

def init_latent_dynamics_params(key: jax.random.PRNGKey,
                                latent_dim: int,
                                hidden_dim: int) -> Dict[str, Any]:
    """Initialize the MLP that defines dz/dt in the latent space."""
    k1, k2 = random.split(key, 2)
    linear1 = init_linear_params(k1, latent_dim, hidden_dim)
    linear2 = init_linear_params(k2, hidden_dim, latent_dim)
    return {"linear1": linear1, "linear2": linear2}

def latent_dynamics_func(params: Dict[str, Any],
                         t: float,
                         z: jnp.ndarray) -> jnp.ndarray:
    """Compute dz/dt using a one-hidden-layer MLP with tanh activation."""
    h = jnp.tanh(linear_forward(params["linear1"], z))
    dz_dt = linear_forward(params["linear2"], h)
    return dz_dt


def conv_node_dynamics_func(params: dict, t: float, x: jnp.ndarray) -> jnp.ndarray:
    """
    Convolutional dynamics function for ConvNODE.

    x is the current state (e.g. an image tensor) of shape (H, W, C).
    This function applies two convolutional layers with a ReLU activation in between.
    """
    h = jax.nn.relu(conv_node_forward(params["conv1"], x, stride=(1, 1), padding="SAME"))
    dx_dt = conv_node_forward(params["conv2"], h, stride=(1, 1), padding="SAME")
    return dx_dt

def init_decoder_params(key: jax.random.PRNGKey,
                        latent_dim: int,
                        hidden_dim: int,
                        output_dim: int) -> Dict[str, Any]:
    """Initialize the MLP decoder that maps z(t) back to x(t)."""
    k1, k2 = random.split(key, 2)
    linear1 = init_linear_params(k1, latent_dim, hidden_dim)
    linear2 = init_linear_params(k2, hidden_dim, output_dim)
    return {"linear1": linear1, "linear2": linear2}

def decode(decoder_params: Dict[str, Any],
           z: jnp.ndarray) -> jnp.ndarray:
    """Decode latent state z into observed space x."""
    h = jnp.tanh(linear_forward(decoder_params["linear1"], z))
    x_recon = linear_forward(decoder_params["linear2"], h)
    return x_recon

def latent_ode_loss(encoder_params: Dict[str, Any],
                    latent_dynamics_params: Dict[str, Any],
                    decoder_params: Dict[str, Any],
                    key: jax.random.PRNGKey,
                    x_seq: jnp.ndarray,
                    t_seq: jnp.ndarray,
                    beta: float) -> jnp.ndarray:
    """
    Compute the reconstruction + KL loss for a single time-series x_seq.
    x_seq: shape (T, obs_dim)
    t_seq: shape (T,) for time points
    """
    # Encode
    z0_mean, z0_logvar = encode(encoder_params, x_seq)
    z0 = reparameterize(key, z0_mean, z0_logvar)

    # Integrate latent dynamics from t0 to t1 at all t_seq
    t0 = t_seq[0]
    t1 = t_seq[-1]
    term = diffrax.ODETerm(lambda t, y, args: latent_dynamics_func(args, t, y))
    solver = diffrax.Dopri5()
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=z0,
        args=latent_dynamics_params,
        adjoint=adjoint,
        saveat=diffrax.SaveAt(ts=t_seq)  # ensure we get z(t) at all t_seq
    )
    z_t = sol.ys  # shape (T, latent_dim)

    # Decode each z(t)
    x_recon = vmap(lambda z: decode(decoder_params, z))(z_t)  # shape (T, obs_dim)
    # Reconstruction loss: (for continuous data, MSE corresponds to a Gaussian likelihood)
    recon_loss = jnp.mean((x_recon - x_seq) ** 2)

    # KL divergence (per time-step average)
    kl_div = -0.5 * jnp.sum(1.0 + z0_logvar - (z0_mean**2) - jnp.exp(z0_logvar))
    kl_div /= x_seq.shape[0]

    return recon_loss + beta * kl_div  # beta can be annealed

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
        """
        :param input_dim: dimension of x(t)
        :param rnn_hidden: hidden units for GRU encoder
        :param latent_dim: dimension of z(t)
        :param dynamics_hidden: hidden units for the MLP defining dz/dt
        :param decoder_hidden: hidden units for the MLP decoder
        :param timesteps: number of time points in training data
        :param lr: learning rate
        :param key: random PRNGKey
        """
        self.input_dim = input_dim
        self.rnn_hidden = rnn_hidden
        self.latent_dim = latent_dim
        self.dynamics_hidden = dynamics_hidden
        self.decoder_hidden = decoder_hidden
        self.timesteps = timesteps
        self.t_seq = jnp.linspace(0.0, 1.0, timesteps)
        self.lr = lr
        self.key = key

        self.key, subkey = random.split(self.key)
        self.encoder_params = init_biencoder_params(subkey, input_dim, rnn_hidden, latent_dim)

        self.key, subkey = random.split(self.key)
        self.latent_dynamics_params = init_latent_dynamics_params(subkey, latent_dim, dynamics_hidden)

        self.key, subkey = random.split(self.key)
        self.decoder_params = init_decoder_params(subkey, latent_dim, decoder_hidden, input_dim)

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
                    x_batch: jnp.ndarray,  # shape (batch, T, obs_dim)
                    t_seq: jnp.ndarray,
                    beta: float) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], optax.OptState, jnp.ndarray]:
        """
        Perform one training step over a batch of time-series data.
        """
        params_tuple = (encoder_params, latent_dynamics_params, decoder_params)

        def loss_fn(params_tuple):
            enc_params, dyn_params, dec_params = params_tuple
            # Split the key into one per sequence in the batch
            batch_keys = random.split(key, x_batch.shape[0])
            # Compute the loss for each sequence in the batch
            losses = vmap(
                lambda x_seq, k: latent_ode_loss(enc_params, dyn_params, dec_params, k, x_seq, t_seq, beta)
            )(x_batch, batch_keys)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params_tuple)
        updates, opt_state = self.optimizer.update(grads, opt_state, params_tuple)
        new_params = optax.apply_updates(params_tuple, updates)
        return new_params[0], new_params[1], new_params[2], opt_state, loss

    def train(self, x_data: jnp.ndarray, num_epochs: int, batch_size: int) -> Tuple[List[float], List[float]]:
        """
        Train the model on x_data: shape (num_samples, T, obs_dim).
        """
        num_samples = x_data.shape[0]
        num_batches = num_samples // batch_size
        loss_history: List[float] = []
        time_history: List[float] = []

        for epoch in range(num_epochs):
            start_time = time.time()

            # Adjust beta: typically, we warm up the KL divergence (e.g., from 0 to 1)
            beta = epoch / num_epochs  # increasing KL weight over time
            # Alternatively, if the paper requires beta to decrease, use: beta = 1.0 - (epoch / num_epochs)

            # Shuffle the data (using a new key for permutation)
            self.key, perm_key = random.split(self.key)
            perm = random.permutation(perm_key, num_samples)
            x_data_shuffled = x_data[perm]
            epoch_loss = 0.0

            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                x_batch = x_data_shuffled[batch_start:batch_end]
                # Update the key for each batch to ensure randomness
                self.key, subkey = random.split(self.key)

                (self.encoder_params,
                 self.latent_dynamics_params,
                 self.decoder_params,
                 self.opt_state,
                 loss) = self._train_step(
                     self.encoder_params,
                     self.latent_dynamics_params,
                     self.decoder_params,
                     self.opt_state,
                     subkey,
                     x_batch,
                     self.t_seq,
                     beta
                 )
                epoch_loss += loss

            epoch_loss /= num_batches
            end_time = time.time()
            epoch_time = end_time - start_time
            loss_history.append(float(epoch_loss))
            time_history.append(epoch_time)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f}s")

        return loss_history, time_history

    def predict(self, x_seq: jnp.ndarray, pred_timesteps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict a trajectory given an initial observed sequence x_seq (T, input_dim).
        pred_timesteps can exceed T for extrapolation.
        Returns (x_pred, t_pred).
        """
        self.key, subkey = random.split(self.key)
        z0_mean, z0_logvar = encode(self.encoder_params, x_seq)
        z0 = reparameterize(subkey, z0_mean, z0_logvar)

        t_pred = jnp.linspace(0.0, 1.0, pred_timesteps)
        term = diffrax.ODETerm(lambda t, y, args: latent_dynamics_func(args, t, y))
        solver = diffrax.Dopri5()
        adjoint = diffrax.RecursiveCheckpointAdjoint()

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(t_pred[0]),
            t1=float(t_pred[-1]),
            dt0=0.01,
            y0=z0,
            args=self.latent_dynamics_params,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(ts=t_pred)
        )
        z_t = sol.ys

        # Decode each z(t)
        x_pred = vmap(lambda z: decode(self.decoder_params, z))(z_t)
        return x_pred, t_pred

class ConvNODE:
    def __init__(self, image_shape: tuple, hidden_channels: int, kernel_size: int = 3, lr: float = 0.001,
                 key: jax.random.PRNGKey = random.PRNGKey(0), solver=None):
        """
        Convolutional Neural ODE Model for image data (non-augmented).

        :param image_shape: Tuple (H, W, C) for the original image.
        :param hidden_channels: Number of channels in the hidden convolution layer.
        :param kernel_size: Kernel size for the convolution layers.
        :param lr: Learning rate (for later use).
        :param key: JAX random key.
        :param solver: diffrax ODE solver to use; defaults to diffrax.Dopri5.
        """
        self.image_shape = image_shape  # (H, W, C)
        self.orig_channels = image_shape[-1]
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.lr = lr
        self.key = key

        # Initialize convolutional dynamics parameters.
        # conv1: from orig_channels -> hidden_channels.
        # conv2: from hidden_channels -> orig_channels.
        key, subkey1, subkey2 = random.split(self.key, 3)
        conv1_shape = (kernel_size, kernel_size, self.orig_channels, hidden_channels)
        conv2_shape = (kernel_size, kernel_size, hidden_channels, self.orig_channels)
        self.dynamics_params = {
            "conv1": conv_node_init_conv_params(subkey1, conv1_shape),
            "conv2": conv_node_init_conv_params(subkey2, conv2_shape)
        }

        self.solver = solver if solver is not None else diffrax.Dopri5()

    def integrate(self, x0: jnp.ndarray, t0: float, t1: float, t_eval: jnp.ndarray) -> jnp.ndarray:
        """
        Integrate the ConvNODE from time t0 to t1 given an initial image x0.

        :param x0: Initial image tensor of shape (H, W, C).
        :param t0: Initial time.
        :param t1: Final time.
        :param t_eval: 1D array of time points at which to save the solution.
        :return: Integrated image trajectories (shape: (len(t_eval), H, W, C)).
        """
        term = diffrax.ODETerm(lambda t, x, args: conv_node_dynamics_func(args, t, x))
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=x0,
            args=self.dynamics_params,
            saveat=diffrax.SaveAt(ts=t_eval)
        )
        # sol.ys shape: (T, H, W, C)
        return sol.ys

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
        :param t_eval: 1D array of time points at which to save the solution.
        :param optimizer: An optax optimizer.
        :param opt_state: The current optimizer state.
        :return: Tuple (loss, new optimizer state)
        """

        def loss_fn(params):
            term = diffrax.ODETerm(lambda t, x, args: conv_node_dynamics_func(args, t, x))
            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                t0=t0,
                t1=t1,
                dt0=0.1,
                y0=x,
                args=params,
                saveat=diffrax.SaveAt(ts=t_eval)
            )
            # Use the final integrated image (at t1) as the prediction.
            x_pred = sol.ys[-1]
            return jnp.mean((x_pred - x) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(self.dynamics_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        self.dynamics_params = optax.apply_updates(self.dynamics_params, updates)
        return loss, new_opt_state





