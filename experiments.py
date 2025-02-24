from typing import Tuple, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from latent_ode_model import LatentODEModel

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

class RNNBaselineGaussian(nn.Module):

    def __init__(self, input_dim : int = 2, time_dim : int = 1, hidden_dim : int = 25, output_dim : int = 2):
        """
        Init model

        :param input_dim: Input dimension
        :param time_dim: Time dimension
        :param hidden_dim: Hidden dimension
        :param output_dim: Output dimension
        """
        super(RNNBaselineGaussian, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * output_dim)

    def forward(self, x : torch.Tensor):
        """
        Forward inference

        :param x: X
        :return: Preds
        """

        out, _ = self.rnn(x)  # out: (batch, timesteps, hidden_dim)
        out = self.fc(out)  # out: (batch, timesteps, 2 * output_dim)
        mu, log_var = out.chunk(2, dim=-1)  # Split into mean and log-variance
        return mu, log_var

class RNNBaselineTimeDiffGaussian(nn.Module):
    """
    A baseline RNN whose inputs are concatenated with time differences.
    """

    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 25, output_dim: int = 2):
        """
        Init model

        :param input_dim: Input dimension
        :param time_dim: Time dimension
        :param hidden_dim: Hidden dimension
        :param output_dim: Output dimension
        """

        super(RNNBaselineTimeDiffGaussian, self).__init__()
        self.rnn = nn.RNN(input_dim + time_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * output_dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward inference

        :param x: X
        :param dt: Time difference
        :return: Preds
        """
        batch_size, timesteps, _ = x.size()
        if dt.dim() == 2:
            dt = dt.unsqueeze(0).expand(batch_size, -1, -1)  # shape (batch, timesteps, 1)

        x_cat = torch.cat([x, dt], dim=2)  # shape (batch, timesteps, input_dim + time_dim)
        out, _ = self.rnn(x_cat)
        out = self.fc(out)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var

def gaussian_nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood loss for a Gaussian with mean=mu, variance=exp(log_var).
    Args:
        mu: shape (batch, timesteps, output_dim)
        log_var: shape (batch, timesteps, output_dim)
        target: shape (batch, timesteps, output_dim)
    Returns:
        Scalar loss (averaged).
    """
    var = torch.exp(log_var)
    nll = 0.5 * (log_var + (target - mu) ** 2 / var)
    return nll.mean()


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


def train_rnn_baseline_gaussian(
    model: nn.Module,
    data: jnp.ndarray,
    num_epochs: int = 20,
    lr: float = 0.001,
    use_time_diff: bool = False
) -> Tuple[List[float], float]:
    """
    Trains a Gaussian-output RNN model on the spiral dataset.
    data: shape (num_spirals, timesteps, 2) as a JAX array
    Returns: (list of epoch losses, total training time)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Convert JAX array to NumPy -> Torch
    data_np = np.array(data)
    batch_size = 32
    losses = []
    start_time = time.time()

    if use_time_diff:
        dt = torch.ones(data_np.shape[1], 1).to(device)  # shape (timesteps, 1)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        perm = np.random.permutation(data_np.shape[0])
        for i in range(0, data_np.shape[0], batch_size):
            indices = perm[i : i + batch_size]
            batch = torch.tensor(data_np[indices], dtype=torch.float32).to(device)
            optimizer.zero_grad()

            if use_time_diff:
                mu, log_var = model(batch, dt)
            else:
                mu, log_var = model(batch)

            loss = gaussian_nll_loss(mu, log_var, batch)  # compare to batch
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= (data_np.shape[0] / batch_size)
        print(f"Gaussian RNN Epoch {epoch+1}/{num_epochs}, NLL Loss: {epoch_loss:.6f}")
        losses.append(epoch_loss)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time for Gaussian RNN baseline: {total_time:.2f} seconds")
    return losses, total_time


def main():
    # Step A: Generate the dataset.
    num_spirals = 1000
    timesteps = 100
    noise_std = 0.1
    data = generate_spiral_dataset(num_spirals, timesteps, noise_std)
    print("Dataset shape:", data.shape)  # Expected: (1000, 100, 2)
    plot_spirals(data, num_samples=5)


    latent_ode_model = LatentODEModel(
        input_dim=2,
        rnn_hidden=25,
        latent_dim=4,
        dynamics_hidden=20,
        decoder_hidden=20,
        timesteps=timesteps,
        lr=0.01
    )
    print("Training Latent ODE model...")
    x_data_jnp = jnp.array(data)
    ode_loss_history, ode_time_history = latent_ode_model.train(x_data_jnp, num_epochs=100, batch_size=32)
    print("Latent ODE model training complete.")
    print("Final Loss:", ode_loss_history[-1])
    print("Total Training Time (approx):", sum(ode_time_history))

    sample_idx = 0
    sample_traj = data[sample_idx]  # shape (100, 2)
    x_pred_fit, _ = latent_ode_model.predict(jnp.array(sample_traj), pred_timesteps=timesteps)
    x_pred_extrap, _ = latent_ode_model.predict(jnp.array(sample_traj), pred_timesteps=150)
    sample_traj_np = np.array(sample_traj)
    x_pred_fit_np = np.array(x_pred_fit)
    x_pred_extrap_np = np.array(x_pred_extrap)
    plot_predictions(sample_traj_np, x_pred_fit_np, title="Latent ODE - Fitting")
    plot_predictions(sample_traj_np, x_pred_extrap_np, title="Latent ODE - Extrapolation")

    print("Training Baseline RNN (without time difference)...")
    rnn_model = RNNBaselineGaussian(input_dim=2, hidden_dim=25, output_dim=2)
    rnn_losses, rnn_time = train_rnn_baseline_gaussian(rnn_model, data, num_epochs=100, lr=0.001, use_time_diff=False)
    rnn_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_torch = torch.tensor(np.array(sample_traj), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, log_var = rnn_model(sample_torch)
        rnn_pred = mu.cpu().numpy()[0]
    plot_predictions(sample_traj_np, rnn_pred, title="Baseline RNN Predictions")

    print("Training Baseline RNN (with time difference)...")
    rnn_model_td = RNNBaselineTimeDiffGaussian(input_dim=2, time_dim=1, hidden_dim=25, output_dim=2)
    rnn_losses_td, rnn_time_td = train_rnn_baseline_gaussian(rnn_model_td, data, num_epochs=100, lr=0.001, use_time_diff=True)
    rnn_model_td.eval()
    dt = torch.ones(data.shape[1], 1)
    with torch.no_grad():
        mu, log_var = rnn_model_td(sample_torch, dt.to(device))
        rnn_pred_td = mu.cpu().numpy()[0]
    plot_predictions(sample_traj_np, rnn_pred_td, title="Baseline RNN (Time Diff) Predictions")


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ode_loss_history, label="Latent ODE")
    plt.plot(rnn_losses, label="RNN Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(["Latent ODE", "RNN Baseline"], [sum(ode_time_history), rnn_time], color=["blue", "orange"])
    plt.ylabel("Time (sec)")
    plt.title("Training Time Comparison")
    plt.tight_layout()
    plt.show()


def experiments_2():
    num_spirals = 1000
    train_timesteps = 100  # training data length
    noise_std = 0.1
    data = generate_spiral_dataset(num_spirals, train_timesteps, noise_std)
    print("Training dataset shape:", data.shape)  # Expected: (1000, 100, 2)

    # Instantiate and train the Latent ODE model
    latent_ode_model = LatentODEModel(
        input_dim=2,
        rnn_hidden=25,
        latent_dim=4,
        dynamics_hidden=20,
        decoder_hidden=20,
        timesteps=train_timesteps,
        lr=0.01
    )
    print("Training Latent ODE model on 100-timestep data...")
    x_data_jnp = jnp.array(data)  # shape (1000, 100, 2)
    ode_loss_history, ode_time_history = latent_ode_model.train(x_data_jnp, num_epochs=100, batch_size=32)
    print("Latent ODE training complete. Final Loss:", ode_loss_history[-1])
    print("Total training time (approx):", sum(ode_time_history), "sec")

    # Use first sample from the dataset for predictions
    sample_traj = data[0]  # shape (100, 2)
    sample_traj_np = np.array(sample_traj)

    # 1) Interpolation: Predict using the same 100 timesteps as training
    x_pred_interp, t_pred_interp = latent_ode_model.predict(jnp.array(sample_traj), pred_timesteps=train_timesteps)
    x_pred_interp_np = np.array(x_pred_interp)

    # 2) Extrapolation: Predict for a longer horizon (e.g., 150 timesteps)
    pred_timesteps = 150
    x_pred_extrap, t_pred_extrap = latent_ode_model.predict(jnp.array(sample_traj), pred_timesteps=pred_timesteps)
    x_pred_extrap_np = np.array(x_pred_extrap)

    # For extrapolation, generate a clean ground truth spiral (noise-free) over 150 timesteps.
    clean_spiral = generate_clean_spiral(pred_timesteps, clockwise=True)  # shape (150, 2)
    clean_spiral_np = np.array(clean_spiral)

    # Plot interpolation (fitting) results: compare noisy training sample (100 timesteps) vs. interpolation.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sample_traj_np[:, 0], sample_traj_np[:, 1], 'o-', label="Noisy Training Data (100 timesteps)")
    plt.plot(x_pred_interp_np[:, 0], x_pred_interp_np[:, 1], 'x--', label="Latent ODE Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolation: Fitting")
    plt.legend()

    # Plot extrapolation results: compare clean ground truth and extrapolated prediction.
    plt.subplot(1, 2, 2)
    plt.plot(sample_traj_np[:, 0], sample_traj_np[:, 1], 'o-', label="Noisy Training Data (first 100 timesteps)")
    plt.plot(clean_spiral_np[:, 0], clean_spiral_np[:, 1], 'k-', label="Clean Ground Truth (150 timesteps)")
    plt.plot(x_pred_extrap_np[:, 0], x_pred_extrap_np[:, 1], 'x--', label="Latent ODE Extrapolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Extrapolation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Now train and evaluate the baseline RNN models.
    print("Training Baseline RNN (Gaussian, without time difference)...")
    rnn_model = RNNBaselineGaussian(input_dim=2, hidden_dim=25, output_dim=2)
    rnn_losses, rnn_time = train_rnn_baseline_gaussian(rnn_model, data, num_epochs=100, lr=0.001, use_time_diff=False)
    rnn_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_torch = torch.tensor(np.array(sample_traj), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, log_var = rnn_model(sample_torch)
        rnn_pred = mu.cpu().numpy()[0]
    # Plot RNN predictions (interpolation)
    plt.figure(figsize=(8, 6))
    plot_predictions(sample_traj_np, rnn_pred, title="Baseline RNN (Gaussian) Predictions - Interpolation")
    plt.show()

    # Compare training losses and times
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ode_loss_history, label="Latent ODE")
    plt.plot(rnn_losses, label="RNN Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(["Latent ODE", "RNN Baseline"], [sum(ode_time_history), rnn_time], color=["blue", "orange"])
    plt.ylabel("Time (sec)")
    plt.title("Training Time Comparison")
    plt.tight_layout()
    plt.show()

    print("Extrapolation experiments complete!")


if __name__ == "__main__":
    main()
    #experiments_2()
