import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

class RNNBaseline(nn.Module):

    def __init__(self, input_dim : int = 2, time_dim : int = 1, hidden_dim : int = 25, output_dim : int = 2):
        """
        Init model

        :param input_dim: Input dimension
        :param time_dim: Time dimension
        :param hidden_dim: Hidden dimension
        :param output_dim: Output dimension
        """
        super(RNNBaseline, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x : torch.Tensor): # TODO: Annotate
        """
        Forward inference

        :param x: X
        :return: Preds
        """

        out, _ = self.rnn(x)
        out = self.fc(out)

        return out

class RNNBaselineTimeDiff(nn.Module):
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

        super(RNNBaselineTimeDiff, self).__init__()
        self.rnn = nn.RNN(input_dim + time_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Forward inference

        :param x: X
        :param dt: Time difference
        :return: Preds
        """
        # x: (batch, timesteps, input_dim)
        batch_size, timesteps, _ = x.size()
        dt_expanded = dt.unsqueeze(0).expand(batch_size, -1, -1)
        x_cat = torch.cat([x, dt_expanded], dim=2)
        out, _ = self.rnn(x_cat)
        out = self.fc(out)
        return out


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


def train_rnn_baseline(model, data, num_epochs : int = 20, lr : float = 0.001, use_time_diff : bool = False):
    """
    Trains baseline RNN model

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    model.train()

    losses = []
    if use_time_diff: # Use dt = 1
        dt = torch.ones(data.shape[1], 1)  # shape (timesteps, 1)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        perm = np.random.permutation(data.shape[0])
        for i in range(0, data.shape[0], 32):
            indices = perm[i : i + 32]
            batch = torch.tensor(np.array(data[indices]), dtype=torch.float32).to(device)
            optimizer.zero_grad()
            if use_time_diff:
                output = model(batch, dt.to(device))
            else:
                output = model(batch)

            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= (data.shape[0] / 32)
        print(f"RNN Baseline Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        losses.append(epoch_loss)
    return losses


def main():
    # Step A: Generate the dataset.
    num_spirals = 1000
    timesteps = 100
    noise_std = 0.1
    data = generate_spiral_dataset(num_spirals, timesteps, noise_std)
    print("Dataset shape:", data.shape)  # Expected: (1000, 100, 2)
    plot_spirals(data, num_samples=5)

    # Step B: Train the Latent ODE Model.
    latent_ode_model = LatentODEModel(
        input_dim=2,
        rnn_hidden=25,
        latent_dim=4,
        dynamics_hidden=20,
        decoder_hidden=20,
        timesteps=timesteps,
        lr=0.001
    )
    print("Training Latent ODE model...")
    x_data_jnp = jnp.array(data)
    ode_loss_history, ode_time_history = latent_ode_model.train(x_data_jnp, num_epochs=20, batch_size=32)
    print("Latent ODE model training complete.")
    print("Final Loss:", ode_loss_history[-1])
    print("Total Training Time (approx):", sum(ode_time_history))

    # Step C: Visualize Latent ODE Predictions.
    sample_idx = 0
    sample_traj = data[sample_idx]  # shape (100, 2)
    x_pred_fit, _ = latent_ode_model.predict(jnp.array(sample_traj), pred_timesteps=timesteps)
    x_pred_extrap, _ = latent_ode_model.predict(jnp.array(sample_traj), pred_timesteps=150)
    sample_traj_np = np.array(sample_traj)
    x_pred_fit_np = np.array(x_pred_fit)
    x_pred_extrap_np = np.array(x_pred_extrap)
    plot_predictions(sample_traj_np, x_pred_fit_np, title="Latent ODE - Fitting")
    plot_predictions(sample_traj_np, x_pred_extrap_np, title="Latent ODE - Extrapolation")

    # Step D: Train Baseline RNN (without time difference).
    print("Training Baseline RNN (without time difference)...")
    rnn_model = RNNBaseline(input_dim=2, hidden_dim=25, output_dim=2)
    rnn_losses, rnn_time = train_rnn_baseline(rnn_model, data, num_epochs=20, lr=0.001, use_time_diff=False)
    rnn_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_torch = torch.tensor(np.array(sample_traj), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        rnn_pred = rnn_model(sample_torch).cpu().numpy()[0]
    plot_predictions(sample_traj_np, rnn_pred, title="Baseline RNN Predictions")

    # Step E: Train Baseline RNN (with time difference).
    print("Training Baseline RNN (with time difference)...")
    rnn_model_td = RNNBaselineTimeDiff(input_dim=2, time_dim=1, hidden_dim=25, output_dim=2)
    rnn_losses_td, rnn_time_td = train_rnn_baseline(rnn_model_td, data, num_epochs=20, lr=0.001, use_time_diff=True)
    rnn_model_td.eval()
    dt = torch.ones(data.shape[1], 1)
    with torch.no_grad():
        rnn_pred_td = rnn_model_td(sample_torch, dt.to(device)).cpu().numpy()[0]
    plot_predictions(sample_traj_np, rnn_pred_td, title="Baseline RNN (Time Diff) Predictions")

    # Step F: Compare Losses and Training Times.
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

    print("All experiments completed!")


if __name__ == "__main__":
    main()
