# img_experiments.py

import time
import jax
import jax.numpy as jnp
from jax import random
import optax
import diffrax
import matplotlib.pyplot as plt
import numpy as np

# Import dataloaders (ensure dataloaders.py is in your PYTHONPATH)
from dataloaders import get_mnist_dataloader

# Import the models and their dynamics functions from augmented_node_model.py
from augmented_node_model import ConvAugmentedNODE
from latent_node_model import ConvNODE


def train_model(model, optimizer, opt_state, train_loader, num_epochs):
    """
    Train a model using its train_step method on all images in the training dataloader.
    Since train_step is defined for a single image, we iterate over each sample.

    :param model: The model instance (ConvNODE or ConvAugmentedNeuralODEModel).
    :param optimizer: An optax optimizer.
    :param opt_state: The initial optimizer state.
    :param train_loader: A dataloader yielding batches (images, labels).
    :param num_epochs: Number of training epochs.
    :return: The final optimizer state and a list of average training losses per epoch.
    """
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for batch in train_loader:
            images, _ = batch
            # Convert images to a JAX array.
            images_j = jnp.array(images)
            # Iterate over images in the batch.
            for i in range(images_j.shape[0]):
                x = images_j[i]  # shape: (H, W, C)
                loss, opt_state = model.train_step(
                    x,
                    t0=0.0,
                    t1=1.0,
                    t_eval=jnp.array([0.0, 1.0]),
                    optimizer=optimizer,
                    opt_state=opt_state
                )
                epoch_loss += loss
                count += 1
        avg_loss = epoch_loss / count
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:.6f}")
        losses.append(avg_loss)
    return opt_state, losses


def evaluate_model(model, test_loader):
    """
    Evaluate a model on the test set.
    For each image, integrate from t=0 to t=1 and compute the reconstruction MSE.

    :param model: The model instance.
    :param test_loader: A dataloader yielding (images, labels) batches.
    :return: Average reconstruction loss, and lists of original and reconstructed images.
    """
    total_loss = 0.0
    count = 0
    originals = []
    reconstructions = []
    t_eval = jnp.array([0.0, 1.0])
    for batch in test_loader:
        images, _ = batch
        images_j = jnp.array(images)
        for i in range(images_j.shape[0]):
            x = images_j[i]
            x_pred = model.integrate(x, t0=0.0, t1=1.0, t_eval=t_eval)[-1]
            loss = jnp.mean((x_pred - x) ** 2)
            total_loss += loss
            count += 1
            originals.append(np.array(x))
            reconstructions.append(np.array(x_pred))
    avg_loss = total_loss / count
    return avg_loss, originals, reconstructions


def main():
    # -------------------------------
    # Data Loading
    # -------------------------------
    batch_size = 64
    num_train_epochs = 5  # Adjust as needed for your experiment.
    train_loader = get_mnist_dataloader(batch_size=batch_size, split="train", shuffle=True)
    test_loader = get_mnist_dataloader(batch_size=16, split="test", shuffle=False)

    # -------------------------------
    # Model Setup
    # -------------------------------
    image_shape = (28, 28, 1)
    key1 = random.PRNGKey(0)
    key2 = random.PRNGKey(42)
    conv_node_model = ConvNODE(image_shape=image_shape, hidden_channels=8, kernel_size=3, lr=0.001, key=key1)
    conv_anode_model = ConvAugmentedNODE(image_shape=image_shape, aug_channels=2, hidden_channels=8,
                                                   kernel_size=3, lr=0.001, key=key2)

    # -------------------------------
    # Optimizers
    # -------------------------------
    optimizer = optax.adam(0.001)
    opt_state_convnode = optimizer.init(conv_node_model.dynamics_params)
    opt_state_convanode = optimizer.init(conv_anode_model.dynamics_params)

    # -------------------------------
    # Training
    # -------------------------------
    print("Training ConvNODE (non-augmented)...")
    opt_state_convnode, losses_convnode = train_model(conv_node_model, optimizer, opt_state_convnode, train_loader,
                                                      num_train_epochs)

    print("\nTraining Conv ANODE (augmented)...")
    opt_state_convanode, losses_convanode = train_model(conv_anode_model, optimizer, opt_state_convanode, train_loader,
                                                        num_train_epochs)

    # Plot training loss curves.
    plt.figure(figsize=(8, 6))
    plt.plot(losses_convnode, label="ConvNODE Loss")
    plt.plot(losses_convanode, label="Conv ANODE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.show()

    # -------------------------------
    # Evaluation
    # -------------------------------
    test_loss_convnode, originals_convnode, recons_convnode = evaluate_model(conv_node_model, test_loader)
    test_loss_convanode, originals_convanode, recons_convanode = evaluate_model(conv_anode_model, test_loader)

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
