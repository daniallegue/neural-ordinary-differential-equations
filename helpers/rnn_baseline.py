# rnn_baseline.py

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, bidirectional=True):
        """
        Encoder using a bidirectional GRU.

        Args:
            input_dim (int): Dimension of the input time series.
            hidden_dim (int): Number of hidden units for the GRU.
            latent_dim (int): Dimension of the latent variable.
            num_layers (int): Number of GRU layers.
            bidirectional (bool): Whether to use a bidirectional GRU.
        """
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        # If bidirectional, the hidden state dimension is doubled.
        hidden_out_dim = hidden_dim * self.num_directions
        self.fc_mu = nn.Linear(hidden_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_out_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            mu (Tensor): Mean of the latent variable distribution.
            logvar (Tensor): Log-variance of the latent variable distribution.
        """
        # x: (batch, seq_len, input_dim)
        out, h = self.gru(x)  # h: (num_layers*num_directions, batch, hidden_dim)
        if self.bidirectional:
            # Concatenate the final hidden states from both directions of the last layer.
            # h[-2] is from the forward GRU and h[-1] from the backward GRU.
            h_final = torch.cat([h[-2], h[-1]], dim=1)  # shape: (batch, 2*hidden_dim)
        else:
            h_final = h[-1]  # shape: (batch, hidden_dim)
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=1):
        """
        Decoder using a GRU that generates an output sequence from a latent state.

        Args:
            latent_dim (int): Dimension of the latent variable.
            hidden_dim (int): Number of hidden units for the GRU.
            output_dim (int): Dimension of the output (reconstructed) time series.
            num_layers (int): Number of GRU layers.
        """
        super(Decoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        """
        Decode latent variable z into a sequence.

        Args:
            z (Tensor): Latent state of shape (batch, latent_dim).
            seq_len (int): Desired output sequence length.

        Returns:
            outputs (Tensor): Reconstructed sequence (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)
        # Initialize the hidden state for GRU from the latent variable.
        hidden = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)  # (1, batch, hidden_dim)
        # Initialize input token as zeros.
        input_step = torch.zeros(batch_size, 1, self.fc_out.out_features, device=z.device)
        outputs = []
        for t in range(seq_len):
            out, hidden = self.gru(input_step, hidden)
            out = self.fc_out(out)  # (batch, 1, output_dim)
            outputs.append(out)
            input_step = out  # Use the output as the next input.
        outputs = torch.cat(outputs, dim=1)
        return outputs


class RNNVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim=None,
                 num_layers=1, bidirectional=True):
        """
        RNN-VAE model that encodes a time series into a latent variable and decodes it.

        Args:
            input_dim (int): Dimension of input time series.
            hidden_dim (int): Number of hidden units in GRU layers.
            latent_dim (int): Dimension of the latent variable.
            output_dim (int): Dimension of the reconstructed time series. If None, set to input_dim.
            num_layers (int): Number of layers in both encoder and decoder GRUs.
            bidirectional (bool): Whether to use a bidirectional GRU in the encoder.
        """
        super(RNNVAE, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers, bidirectional)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, num_layers)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent variable z from distribution defined by mu and logvar.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass: encode input, sample latent variable, and decode to reconstruct sequence.

        Args:
            x (Tensor): Input tensor (batch, seq_len, input_dim)

        Returns:
            x_recon (Tensor): Reconstructed sequence (batch, seq_len, output_dim)
            mu (Tensor): Mean from encoder.
            logvar (Tensor): Log-variance from encoder.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        seq_len = x.size(1)
        x_recon = self.decoder(z, seq_len)
        return x_recon, mu, logvar

    def initialize_weights(self, init_func=None):
        """
        Initialize model parameters using a provided function.

        Args:
            init_func (callable, optional): A function to initialize weights.
                If None, default initialization is used.
        """
        if init_func is not None:
            self.apply(init_func)

__all__ = ['RNNVAE']
