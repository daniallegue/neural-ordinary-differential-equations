import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time

from rnn_baseline import RNNVAE
from latent_node_model import LatentODEModel

def generate_spiral_dataset(num_spirals : int = 1000, timesteps : int = 100, noise_std : float = 0.1) -> jnp.ndarray:
    """
    Generates spiral dataset

    :param num_spirals: Number of spirals
    :param timesteps: Number of timesteps
    :param noise_std: Gaussian Noise STD
    :return: Dataset
    """

    data = []
    t = jnp.linspace(0, 4 * jnp.pi, timesteps)
    for i in range(num_spirals):
        # Half of the samples clockwise
        if i < num_spirals / 2:
            theta = -t
        else:
            theta = t

        r = t / (4 * jnp.pi)
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        spiral = jnp.stack([x, y], axis = 1) # s: (timesteps, 2)
        spiral += np.random.randn(timesteps, 2) * noise_std

        data.append(spiral)


    data = jnp.array(data)
    return data


def generate_clean_spiral(timesteps: int, clockwise: bool = True) -> jnp.ndarray:
    """
    Generates a (noise-free) spiral for extrapolation evaluation.
    For extrapolation, we extend the time horizon from 0 to 6π.

    :param timesteps: Number of timesteps for the clean spiral.
    :param clockwise: If True, generate a clockwise spiral; otherwise, counter-clockwise.
    :return: A JAX array of shape (timesteps, 2).
    """

    t = jnp.linspace(0, 6 * jnp.pi, timesteps)
    if clockwise:
        theta = -t
    else:
        theta = t


    r = t / (6 * jnp.pi)
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    spiral = jnp.stack([x, y], axis=1)
    return spiral

def plot_spirals(data, num_samples=5):
    """
    Plots several sample spirals.
    """
    plt.figure(figsize=(8, 6))
    for i in range(num_samples):
        plt.plot(data[i, :, 0], data[i, :, 1], marker='o')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sample Spirals")
    plt.show()


def plot_predictions(true_data, pred_data, title="Predictions vs True"):
    """
    Plots true and predicted trajectories.
    true_data and pred_data: numpy arrays of shape (T, 2)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(true_data[:, 0], true_data[:, 1], 'o-', label="True")
    plt.plot(pred_data[:, 0], pred_data[:, 1], 'x--', label="Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    # -------------------------------
    # Generate the Spiral Dataset
    # -------------------------------
    num_spirals = 1000
    timesteps = 100
    noise_std = 0.1
    data_jax = generate_spiral_dataset(num_spirals, timesteps, noise_std)
    # Convert JAX array to NumPy for plotting and PyTorch training where needed.
    data_np = np.array(data_jax)
    print("Generated spiral dataset.")
    plot_spirals(data_np, num_samples=5)

    # -------------------------------
    # Train the Latent ODE Model (JAX)
    # -------------------------------
    from jax import random
    key = random.PRNGKey(0)
    latent_ode_model = LatentODEModel(
        input_dim=2,
        rnn_hidden=32,
        latent_dim=6,
        dynamics_hidden=20,
        decoder_hidden=32,
        timesteps=timesteps,
        lr=0.001,
        key=key
    )
    num_epochs = 100
    batch_size = 32
    print("Training Latent ODE model...")
    ode_loss_history, ode_time_history = latent_ode_model.train(data_jax, num_epochs, batch_size)

    # -------------------------------
    # Train the RNN-VAE Model (PyTorch)
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_vae = RNNVAE(input_dim=2, hidden_dim=32, latent_dim=6, output_dim=2, num_layers=1, bidirectional=True)
    rnn_vae = rnn_vae.to(device)
    optimizer = torch.optim.Adam(rnn_vae.parameters(), lr=0.001)
    rnn_epochs = 100
    rnn_batch_size = 32
    rnn_loss_history = []
    rnn_time_history = []
    # Convert training data to torch tensor
    data_torch = torch.tensor(data_np, dtype=torch.float32).to(device)
    num_samples = data_torch.shape[0]
    num_batches = num_samples // rnn_batch_size

    print("Training RNN-VAE model...")
    for epoch in range(rnn_epochs):
        start_time = time.time()
        perm = torch.randperm(num_samples)
        data_torch = data_torch[perm]
        epoch_loss = 0.0
        for i in range(num_batches):
            batch = data_torch[i * rnn_batch_size : (i + 1) * rnn_batch_size]
            optimizer.zero_grad()
            x_recon, mu, logvar = rnn_vae(batch)
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(x_recon, batch, reduction='mean')
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # Beta annealing schedule: increase KL weight over epochs
            beta = epoch / rnn_epochs
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= num_batches
        epoch_time = time.time() - start_time
        rnn_loss_history.append(epoch_loss)
        rnn_time_history.append(epoch_time)
        print(f"RNN-VAE Epoch {epoch+1}/{rnn_epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f}s")

    # -------------------------------
    # Plot Loss Curves and Training Times
    # -------------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ode_loss_history, label="Latent ODE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Latent ODE Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rnn_loss_history, label="RNN-VAE Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RNN-VAE Training Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ode_time_history, label="Latent ODE Training Time")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Latent ODE Training Times")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rnn_time_history, label="RNN-VAE Training Time", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("RNN-VAE Training Times")
    plt.legend()
    plt.show()

    # Generate a clean spiral (extrapolation: extend to 6π)
    clean_spiral = generate_clean_spiral(timesteps=150, clockwise=True)
    clean_spiral_np = np.array(clean_spiral)
    # For Latent ODE: use first 100 timesteps as observed and predict 150 steps
    observed = clean_spiral[:100]
    ode_pred, t_pred = latent_ode_model.predict(observed, pred_timesteps=150)
    ode_pred = np.array(ode_pred)
    plot_predictions(clean_spiral_np, ode_pred, title="Latent ODE Extrapolation")

    # For RNN-VAE: use first 100 timesteps as observed; for extrapolation, decode for 150 timesteps.
    observed_torch = torch.tensor(np.array(observed)[None, :, :], dtype=torch.float32).to(device)
    rnn_vae.eval()
    with torch.no_grad():
        # Get latent representation
        mu, logvar = rnn_vae.encoder(observed_torch)
        z = rnn_vae.reparameterize(mu, logvar)
        # Decode for 150 timesteps
        rnn_extrap = rnn_vae.decoder(z, seq_len=150).cpu().numpy()[0]
    plot_predictions(clean_spiral_np, rnn_extrap, title="RNN-VAE Extrapolation")


    sample_idx = 0
    sample_spiral = data_np[sample_idx]  # shape (100, 2)
    # Latent ODE interpolation (predict with 100 timesteps)
    ode_interp, _ = latent_ode_model.predict(jnp.array(sample_spiral), pred_timesteps=100)
    ode_interp = np.array(ode_interp)
    # RNN-VAE interpolation: simply run the forward pass.
    sample_torch = torch.tensor(sample_spiral[None, :, :], dtype=torch.float32).to(device)
    rnn_vae.eval()
    with torch.no_grad():
        rnn_interp, _, _ = rnn_vae(sample_torch)
    rnn_interp = rnn_interp.cpu().numpy()[0]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sample_spiral[:, 0], sample_spiral[:, 1], 'o-', label="Ground Truth")
    plt.plot(ode_interp[:, 0], ode_interp[:, 1], 'x--', label="Latent ODE")
    plt.title("Latent ODE Interpolation")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sample_spiral[:, 0], sample_spiral[:, 1], 'o-', label="Ground Truth")
    plt.plot(rnn_interp[:, 0], rnn_interp[:, 1], 'x--', label="RNN-VAE")
    plt.title("RNN-VAE Interpolation")
    plt.legend()
    plt.show()

    # -------------------------------
    # Extrapolation Evaluation: Counter-Clockwise Spiral (Additional Case)
    # -------------------------------
    # Generate a counter-clockwise clean spiral (extrapolation: extend to 6π)
    clean_spiral_ccw = generate_clean_spiral(timesteps=150, clockwise=False)
    clean_spiral_ccw_np = np.array(clean_spiral_ccw)
    # For Latent ODE: use first 100 timesteps as observed and predict 150 steps
    observed_ccw = clean_spiral_ccw[:100]
    ode_pred_ccw, t_pred_ccw = latent_ode_model.predict(observed_ccw, pred_timesteps=150)
    ode_pred_ccw = np.array(ode_pred_ccw)
    plot_predictions(clean_spiral_ccw_np, ode_pred_ccw, title="Latent ODE Extrapolation (Counter-Clockwise)")

    # For RNN-VAE: use first 100 timesteps as observed; decode for 150 timesteps.
    observed_ccw_torch = torch.tensor(np.array(observed_ccw)[None, :, :], dtype=torch.float32).to(device)
    rnn_vae.eval()
    with torch.no_grad():
        mu_ccw, logvar_ccw = rnn_vae.encoder(observed_ccw_torch)
        z_ccw = rnn_vae.reparameterize(mu_ccw, logvar_ccw)
        rnn_extrap_ccw = rnn_vae.decoder(z_ccw, seq_len=150).cpu().numpy()[0]
    plot_predictions(clean_spiral_ccw_np, rnn_extrap_ccw, title="RNN-VAE Extrapolation (Counter-Clockwise)")

if __name__ == "__main__":
    main()