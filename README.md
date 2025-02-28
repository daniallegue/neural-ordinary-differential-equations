# Neural ODE & Augmented Neural ODE Project

> **References**:  
> - Chen, Ricky T. Q., et al. "**Neural Ordinary Differential Equations**." *Advances in Neural Information Processing Systems*, 2018. [[Paper]](https://arxiv.org/abs/1806.07366)  
> - Dupont, Emilien, et al. "**Augmented Neural ODEs**." *Advances in Neural Information Processing Systems*, 2019. [[Paper]](https://arxiv.org/abs/1904.01681)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [ODE Architecture](#2-ode-architecture)  
   - [Ordinary Differential Equation Perspective](#21-ordinary-differential-equation-perspective)  
   - [Similarity to Discrete ResNets](#22-similarity-to-discrete-resnets)  
   - [Why They Can Be Superior](#23-why-they-can-be-superior)  
3. [Implementation](#3-implementation)  
   - [Project Structure](#31-project-structure)  
   - [Key Files](#32-key-files)  
4. [Training Procedure](#4-training-procedure)  
   - [Experiments Folder](#41-experiments-folder)  
   - [Usage Instructions](#42-usage-instructions)  
5. [JAX, JIT, and Diffrax](#5-jax-jit-and-diffrax)

---

## 1. Overview

This project implements two variants of continuous-depth models for learning dynamics from data:

- **Neural ODE**: Interprets a residual network as the discretization of an ODE, then uses a continuous ODE solver to evolve hidden states.  
- **Augmented Neural ODE**: Extends Neural ODEs by increasing the dimensionality of the hidden state, allowing the model to capture more complex dynamics and mitigate expressivity issues.

These approaches differ from conventional deep neural networks in that they treat layers as time steps of an ODE solver, enabling adaptive computation and continuous-time modeling.

---

## 2. ODE Architecture

### 2.1 Ordinary Differential Equation Perspective

In a Neural ODE, the hidden state \(h(t)\) is defined by an ODE:

\[
\frac{dh}{dt} = f(h(t), t; \theta),
\]

where \(f\) is a learnable function (e.g., an MLP or CNN). The output at time \(t_1\) is computed by integrating from \(t_0\) to \(t_1\). In the **Augmented** version, the hidden state is padded with additional dimensions, improving the model’s ability to learn complex dynamics.

### 2.2 Similarity to Discrete ResNets

Discrete ResNets perform updates like:

\[
h_{n+1} = h_n + f(h_n, \theta_n).
\]

This can be viewed as a single Euler step of an ODE. Neural ODEs generalize this idea to a continuous limit, allowing for:

- **Adaptive time steps**: The solver can refine steps as needed.  
- **Fewer parameters** if the solver reuses the same function \(f\).  
- **Natural handling of irregular time sampling** in time-series data.

### 2.3 Why They Can Be Superior

According to the cited papers, Neural ODEs (and their augmented variants) can be superior in:

1. **Memory Efficiency**: Gradients can be computed via the adjoint method without storing intermediate activations.  
2. **Handling Irregular Data**: Continuous formulation seamlessly models non-uniformly sampled time series.  
3. **Expressivity (Augmented ODE)**: Padding the state with extra channels addresses limitations where standard Neural ODEs fail to learn certain topological structures.

---

## 3. Implementation

### 3.1 Project Structure

```
.
├── experiments
│   ├── dataloaders.py
│   ├── img_experiments.py
│   └── time_series_experiments.py
├── helpers
│   └── rnn_baseline.py
└── nodes
    ├── augmented_node_model.py
    ├── latent_node_model.py
    ├── neural_ode_1_step.py
    └── neural_ode_mlp.py
```

### 3.2 Key Files

1. **`experiments/`**  
   - `dataloaders.py`: Utility functions to load and preprocess data (e.g., MNIST or time-series).  
   - `img_experiments.py`: Trains and compares the non-augmented (Neural ODE) and augmented models on image datasets.  
   - `time_series_experiments.py`: Demonstrates usage on time-series data.

2. **`helpers/rnn_baseline.py`**  
   - A baseline RNN-based model for time-series comparisons.

3. **`nodes/`**  
   - `latent_node_model.py`: Implements a basic Neural ODE with latent MLP dynamics.  
   - `augmented_node_model.py`: Implements the augmented version, supporting MLP and convolutional dynamics.  
   - `neural_ode_1_step.py` / `neural_ode_mlp.py`: Additional modules or step-by-step neural ODE variations.

---

## 4. Training Procedure

### 4.1 Experiments Folder

- **`img_experiments.py`**:  
  1. Loads an image dataset (e.g. MNIST) via `dataloaders.py`.  
  2. Creates both the Neural ODE and Augmented ODE models.  
  3. Trains them with a batch-based procedure, logging loss curves.  
  4. Evaluates reconstruction or classification performance.

- **`time_series_experiments.py`**:  
  1. Loads or generates time-series data.  
  2. Compares RNN-based baselines to Neural ODE approaches.  
  3. Plots predictions vs. ground truth.

### 4.2 Usage Instructions

1. **Install Dependencies**  
   ```bash
   pip install jax jaxlib diffrax optax tensorflow-datasets matplotlib numpy
   ```

2. **Run Image Experiments**  
   ```bash
   python experiments/img_experiments.py
   ```

3. **Run Time-Series Experiments**  
   ```bash
   python experiments/time_series_experiments.py
   ```

4. **Adjust Hyperparameters**  
   - Modify model constructors in `latent_node_model.py` or `augmented_node_model.py`.  
   - Tweak learning rates or solver tolerances in the experiment scripts.

---

## 5. JAX, JIT, and Diffrax

- **JAX**: Provides auto-differentiation and composable function transformations (like `jit`). This allows us to efficiently compute gradients of ODE solutions without storing intermediate states (via the adjoint method).

- **JIT Compilation**: By decorating key functions (e.g., the batch training step) with `@jax.jit`, we compile them to XLA, leading to significant speedups in both forward and backward passes.

- **Diffrax**: A library for numerical integration in JAX. It supplies ODE solvers (e.g. `Dopri5`, `Tsit5`) that operate on JAX arrays, enabling end-to-end differentiability and advanced features like checkpointing for memory efficiency.

By combining these tools, we can implement continuous-time models in a Pythonic, high-performance manner, bridging the gap between discrete deep networks and continuous dynamical systems.
