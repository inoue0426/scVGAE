import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from sklearn.metrics.pairwise import pairwise_kernels
from torch.nn import BatchNorm1d, CrossEntropyLoss, Dropout, Linear, Module, MSELoss
from torch.nn.functional import relu, softplus
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GraphNorm
from torch_sparse import SparseTensor
from tqdm import tqdm


def get_topX(X):
    """Retain similarities above the 85th percentile."""
    return X * np.array(X > np.percentile(X, 85), dtype=int)


def get_adj(x):
    """Create a binary sparse adjacency matrix from non-zero similarities."""
    adj = SparseTensor(
        row=torch.tensor(np.array(x.nonzero()))[0],
        col=torch.tensor(np.array(x.nonzero()))[1],
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    return adj


def get_data(X, metric="linear"):
    """Create the cell feature matrix and thresholded cell-cell graph."""
    dist = pairwise_kernels(X, metric=metric)
    dist_x = get_topX(dist)
    return torch.tensor(X.values, dtype=torch.float), get_adj(dist_x)


def ZINBLoss(y_true, y_pred, theta, pi, eps=1e-10):
    """Compute the zero-inflated negative binomial loss.

    Parameters
    ----------
    y_true : torch.Tensor
        Observed expression matrix.
    y_pred : torch.Tensor
        Predicted ZINB mean parameter (mu).
    theta : torch.Tensor
        Predicted dispersion parameter.
    pi : torch.Tensor
        Predicted zero-inflation probability.
    eps : float
        Small constant for numerical stability.
    """

    # Negative-binomial negative log-likelihood for non-zero observations.
    nb_terms = (
        -torch.lgamma(y_true + theta)
        + torch.lgamma(y_true + 1)
        + torch.lgamma(theta)
        - theta * torch.log(theta + eps)
        + theta * torch.log(theta + y_pred + eps)
        - y_true * torch.log(y_pred + theta + eps)
        + y_true * torch.log(y_pred + eps)
    )

    is_zero = (y_true < eps).float()
    zero_terms = torch.log(
        pi + (1 - pi) * torch.pow(1 + y_pred / theta, -theta) + eps
    )

    result = -torch.sum(zero_terms * is_zero + (1 - is_zero) * nb_terms)
    return torch.round(result, decimals=3)


def compute_loss(x_original, x_recon, z_mean, z_dropout, z_dispersion, alpha):
    """Compute the historical scVGAE objective.

    The experiments use:
        alpha * ZINB loss + (1 - alpha) * reconstruction MSE.
    """
    zinb_loss = ZINBLoss(x_original, z_mean, z_dispersion, z_dropout)
    mse_loss = MSELoss()(x_recon, x_original)
    total_loss = alpha * zinb_loss + (1 - alpha) * mse_loss
    return total_loss


class VGAE(Module):
    """ZINB-based graph autoencoder used by scVGAE.

    Note: the historical class name is retained for backward compatibility.
    The architecture does not perform variational latent sampling or use a KL term.
    """

    def __init__(self, params):
        super(VGAE, self).__init__()

        self.dropout1 = nn.Dropout(params["dropout1"])
        self.dropout2 = nn.Dropout(params["dropout2"])

        # Graph encoder and ZINB parameter heads.
        self.gcn1 = GCNConv(params["input_dim"], params["hidden1"])
        self.gn1 = GraphNorm(params["hidden1"])
        self.gcn2_mean = GCNConv(params["hidden1"], params["input_dim"])
        self.gcn2_dropout = GCNConv(params["hidden1"], params["input_dim"])
        self.gcn2_dispersion = GCNConv(params["hidden1"], params["input_dim"])

        # Decoder with two linear layers.
        self.fc1 = Linear(params["input_dim"], params["hidden2"])
        self.bn2 = BatchNorm1d(params["hidden2"])
        self.fc2 = Linear(params["hidden2"], params["input_dim"])

        self.batch_norm1 = BatchNorm1d(params["input_dim"])
        self.batch_norm2 = BatchNorm1d(params["hidden0"])

    def encode(self, x, adj):
        x = relu(self.gn1(self.gcn1(x, adj)))
        x = self.dropout1(x)

        z_mean = torch.exp(self.gcn2_mean(x, adj.t()))
        z_dropout = torch.sigmoid(self.gcn2_dropout(x, adj.t()))
        z_dispersion = torch.exp(self.gcn2_dispersion(x, adj.t()))
        return z_mean, z_dropout, z_dispersion

    def decode(self, z):
        z = relu(self.bn2(self.fc1(z)))
        z = self.dropout2(z)
        return relu(self.fc2(z))

    def forward(self, x, adj, x_t):
        z_mean, z_dropout, z_dispersion = self.encode(x, adj.t())
        x_recon = self.decode(z_mean) + self.batch_norm1(x) + self.batch_norm2(x_t).T
        return x_recon, z_mean, z_dropout, z_dispersion


def run_model(input_data, verbose=False, device=False):
    """Run scVGAE on a cell-by-gene expression matrix."""

    params = {
        "dropout1": 0.2,
        "dropout2": 0.4,
        "epochs": 100,
        "hidden1": 128,
        "hidden2": 1024,
        "lr": 0.0001,
        "alpha": 0.05,
    }

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, adj = get_data(input_data)
    x_t = torch.tensor(input_data.T.values, dtype=torch.float)

    x = x.to(device)
    adj = adj.to(device)
    x_t = x_t.to(device)

    params["input_dim"] = input_data.shape[1]
    params["hidden0"] = input_data.shape[0]

    model = VGAE(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    losses = []

    if verbose:
        epochs = tqdm(range(params["epochs"]))
    else:
        epochs = range(params["epochs"])

    for epoch in epochs:
        x_recon, z_mean, z_dropout, z_dispersion = model(x, adj, x_t)

        # Keep the historical loss weighting while passing ZINB parameters
        # in their intended order: mean, dropout probability, dispersion.
        loss = compute_loss(
            x,
            x_recon,
            z_mean,
            z_dropout,
            z_dispersion,
            params["alpha"],
        ).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    pred = x_recon.cpu().detach().numpy()
    return pred
