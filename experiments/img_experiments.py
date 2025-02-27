# img_experiments.py

import time
import jax.numpy as jnp
from jax import random
import jax
import optax
import matplotlib.pyplot as plt
import numpy as np

from dataloaders import get_mnist_dataloader

# Import the nodes and their dynamics functions from augmented_node_model.py
from nodes.augmented_node_model import ConvAugmentedNODE
from nodes.latent_node_model import ConvNODE


def train_model(model, optimizer, opt_state, train_loader_fn, num_epochs):
    """
    Train a model using its batch_train_step method on all images in the training dataloader.
    The batch_train_step computes the loss and updates the model's parameters on a whole batch.

    :param model: The model instance (ConvNODE or ConvAugmentedNeuralODEModel).
    :param optimizer: An optax optimizer.
    :param opt_state: The initial optimizer state.
    :param train_loader_fn: A function that returns a new iterator over training batches.
    :param num_epochs: Number of training epochs.
    :return: The final optimizer state and lists of average training losses, NFEs, and epoch times.
    """
    losses_per_epoch = []
    times_per_epoch = []

    for epoch in range(num_epochs):
        print("Started epoch:", epoch)
        start_time = time.time()

        # Reinitialize the training iterator each epoch.
        train_loader = train_loader_fn()

        epoch_loss = 0.0
        epoch_nfe = 0  # sum of NFEs in this epoch
        count = 0

        for batch in train_loader:
            print(len(batch))
            images, _ = batch
            images_j = jnp.array(images)  # shape: (B, H, W, C)
            # Call the batch_train_step once per batch.
            new_params, loss, opt_state = model.batch_train_step(
                model.dynamics_params,
                images_j,
                t0=0.0,
                t1=1.0,
                t_eval=jnp.array([0.0, 1.0]),
                solver=model.solver,
                optimizer=optimizer,
                opt_state=opt_state
            )
            model.dynamics_params = new_params

            batch_size = images_j.shape[0]
            epoch_loss += loss * batch_size
            count += batch_size

        avg_loss = epoch_loss / count if count > 0 else float('nan')

        losses_per_epoch.append(avg_loss)

        epoch_time = time.time() - start_time
        times_per_epoch.append(epoch_time)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss={avg_loss:.6f}, Time={epoch_time:.2f}s")

    return opt_state, losses_per_epoch, times_per_epoch


def evaluate_model(model, test_loader_fn):
    """
    Evaluate a model on the test set.
    For each batch, integrate from t=0 to t=1 in a vectorized manner and compute the reconstruction MSE.

    :param model: The model instance.
    :param test_loader_fn: A function that returns a new iterator over test batches.
    :return: Average reconstruction loss, and lists of original and reconstructed images.
    """
    total_loss = 0.0
    count = 0
    originals = []
    reconstructions = []
    t_eval = jnp.array([0.0, 1.0])
    test_loader = test_loader_fn()

    # Disable JIT during eval
    with jax.disable_jit():
        for batch in test_loader:
            images, _ = batch
            images_j = jnp.array(images)  # Expected shape: (B, H, W) or (B, H, W, C)
            if images_j.ndim == 3:
                images_j = images_j[..., None]  # Now shape is (B, H, W, 1)
            # Vectorize integration over the batch.
            batch_integrated = jax.vmap(lambda x: model.integrate(x, t0=0.0, t1=1.0, t_eval=t_eval)[-1])(images_j)
            if batch_integrated.ndim == 3:
                batch_integrated = batch_integrated[..., None]
            loss = jnp.mean((batch_integrated - images_j) ** 2)
            total_loss += loss * images_j.shape[0]
            count += images_j.shape[0]
            # Instead of appending the whole batch, iterate over the batch dimension.
            for i in range(images_j.shape[0]):
                originals.append(np.array(images_j[i]))
                reconstructions.append(np.array(batch_integrated[i]))
        avg_loss = total_loss / count if count > 0 else float('nan')
    return avg_loss, originals, reconstructions


def plot_results(losses_convnode, losses_convanode,
                 nfes_convnode=None, nfes_convanode=None,
                 times_convnode=None, times_convanode=None):
    """
    Plots training curves with a distinct line style (border around markers),
    plus optional NFE and time plots if provided.
    """

    epochs = np.arange(1, len(losses_convnode) + 1)

    # 1) Plot Loss vs. Epoch
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses_convnode, label="ConvNODE",
             linewidth=2, marker='o', markeredgecolor='k',
             markerfacecolor='none', markersize=6)
    plt.plot(epochs, losses_convanode, label="ConvAugmentedNODE",
             linewidth=2, marker='s', markeredgecolor='k',
             markerfacecolor='none', markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # 2) Plot NFE vs. Epoch (only if nfes_* are provided)
    # if nfes_convnode is not None and nfes_convanode is not None:
    #     plt.figure(figsize=(6, 4))
    #     plt.plot(epochs, nfes_convnode, label="ConvNODE",
    #              linewidth=2, marker='o', markeredgecolor='k',
    #              markerfacecolor='none', markersize=6)
    #     plt.plot(epochs, nfes_convanode, label="ConvAugmentedNODE",
    #              linewidth=2, marker='s', markeredgecolor='k',
    #              markerfacecolor='none', markersize=6)
    #     plt.xlabel("Epoch")
    #     plt.ylabel("NFE")
    #     plt.title("Number of Function Evaluations vs. Epoch")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # 3) Plot Time vs. Epoch (only if times_* are provided)
    if times_convnode is not None and times_convanode is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, times_convnode, label="ConvNODE",
                 linewidth=2, marker='o', markeredgecolor='k',
                 markerfacecolor='none', markersize=6)
        plt.plot(epochs, times_convanode, label="ConvAugmentedNODE",
                 linewidth=2, marker='s', markeredgecolor='k',
                 markerfacecolor='none', markersize=6)
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.title("Training Time vs. Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Data
    batch_size = 128
    num_train_epochs = 5
    train_loader_fn = lambda: get_mnist_dataloader(batch_size=batch_size, split="train", shuffle=True, limit=1000)

    test_loader_fn = lambda: get_mnist_dataloader(batch_size=64, split="test", shuffle=False, limit=200)

    # Models
    image_shape = (28, 28, 1)
    key1 = random.PRNGKey(0)
    key2 = random.PRNGKey(42)
    conv_node_model = ConvNODE(image_shape=image_shape, hidden_channels=8, kernel_size=3, lr=0.001, key=key1)
    conv_anode_model = ConvAugmentedNODE(image_shape=image_shape, aug_channels=2, hidden_channels=8,
                                         kernel_size=3, lr=0.001, key=key2)

    # Optimizers
    optimizer_conv = optax.adam(0.001)
    optimizer_aconv = optax.adam(0.001)
    opt_state_convnode = optimizer_conv.init(conv_node_model.dynamics_params)
    opt_state_convanode = optimizer_aconv.init(conv_anode_model.dynamics_params)

    # Training
    print("Training ConvNODE...")
    opt_state_convnode, losses_convnode, times_convnode = train_model(
        conv_node_model, optimizer_conv, opt_state_convnode, train_loader_fn, num_train_epochs
    )

    print("\nTraining ConvAugmentedNODE...")
    opt_state_convanode, losses_convanode, times_convanode = train_model(
        conv_anode_model, optimizer_aconv, opt_state_convanode, train_loader_fn, num_train_epochs
    )

    # Plot the results
    plot_results(losses_convnode, losses_convanode,
                 times_convnode, times_convanode)

    # -------------------------------
    # Evaluation
    # -------------------------------
    test_loss_convnode, originals_convnode, recons_convnode = evaluate_model(conv_node_model, test_loader_fn)
    test_loss_convanode, originals_convanode, recons_convanode = evaluate_model(conv_anode_model, test_loader_fn)

    print("Test Reconstruction Loss (ConvNODE):", test_loss_convnode)
    print("Test Reconstruction Loss (Conv ANODE):", test_loss_convanode)

    # -------------------------------
    # Inference Visualization
    # -------------------------------
    num_display = 8
    fig, axes = plt.subplots(3, num_display, figsize=(num_display * 2, 6))
    for i in range(num_display):
        # Display original image.
        axes[0, i].imshow(originals_convnode[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")
        # Reconstruction from ConvNODE.
        axes[1, i].imshow(recons_convnode[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("ConvNODE")
        # Reconstruction from Conv ANODE.
        axes[2, i].imshow(recons_convanode[i].squeeze(), cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Conv ANODE")
    plt.suptitle("Test Image Reconstructions")
    plt.show()


if __name__ == "__main__":
    main()
