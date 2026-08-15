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
    # Retain similarities above the 85th percentile.
    return X * np.array(X > np.percentile(X, 85), dtype=int)


def get_adj(x):
    adj = SparseTensor(
        row=torch.tensor(np.array(x.nonzero()))[0],
        col=torch.tensor(np.array(x.nonzero()))[1],
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    return adj


def get_data(X, metric="linear"):
    dist = pairwise_kernels(X, metric=metric)
    dist_x = get_topX(dist)
    return torch.tensor(X.values, dtype=torch.float), get_adj(dist_x)


def ZINBLoss(y_true, y_pred, theta, pi, eps=1e-10):
    """
    Compute the ZINB Loss.

    y_true: Ground truth data.
    y_pred: Predicted ZINB mean parameter (mu).
    theta: Predicted dispersion parameter.
    pi: Predicted zero-inflation probability.
    eps: Small constant to prevent log(0).
    """

    # Negative Binomial Loss
    nb_terms = (
        -torch.lgamma(y_true + theta)
        + torch.lgamma(y_true + 1)
        + torch.lgamma(theta)
        - theta * torch.log(theta + eps)
        + theta * torch.log(theta + y_pred + eps)
        - y_true * torch.log(y_pred + theta + eps)
        + y_true * torch.log(y_pred + eps)
    )

    # Zero-Inflation
    zero_inflated = torch.log(pi + (1 - pi) * torch.pow(1 + y_pred / theta, -theta))

    result = -torch.sum(
        torch.log(pi + (1 - pi) * torch.pow(1 + y_pred / theta, -theta))
        * (y_true < eps).float()
        + (1 - (y_true < eps).float()) * nb_terms
    )

    return torch.round(result, decimals=3)


def compute_loss(x_original, x_recon, z_mean, z_dropout, z_dispersion, alpha):
    """
    Compute the historical scVGAE objective:
    alpha * ZINB Loss + (1 - alpha) * MSE Loss.

    Parameters:
    - x_original: Original data matrix.
    - x_recon: Reconstructed matrix from the model.
    - z_mean: ZINB mean parameter.
    - z_dropout: ZINB zero-inflation probability.
    - z_dispersion: ZINB dispersion parameter.
    - alpha: Weight for ZINB loss; (1-alpha) weights MSE loss.

    Returns:
    - total_loss: Combined loss value.
    """

    zinb_loss = ZINBLoss(x_original, z_mean, z_dispersion, z_dropout)
    mse_loss = MSELoss()(x_recon, x_original)
    total_loss = alpha * zinb_loss + (1 - alpha) * mse_loss

    return total_loss


class VGAE(Module):
    def __init__(self, params):
        super(VGAE, self).__init__()

        self.dropout1 = nn.Dropout(params["dropout1"])
        self.dropout2 = nn.Dropout(params["dropout2"])

        # Graph encoder with ZINB parameter heads.
        # The historical class name VGAE is retained for compatibility;
        # this implementation does not perform variational latent sampling.
        self.gcn1 = GCNConv(params["input_dim"], params["hidden1"])
        self.gn1 = GraphNorm(params["hidden1"])
        self.gcn2_mean = GCNConv(params["hidden1"], params["input_dim"])
        self.gcn2_dropout = GCNConv(params["hidden1"], params["input_dim"])
        self.gcn2_dispersion = GCNConv(params["hidden1"], params["input_dim"])

        # Decoder with 2 Linear layers
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

    def forward(
        self,
        x,
        adj,
        x_t,
        adj_t,
    ):
        z_mean, z_dropout, z_dispersion = self.encode(x, adj.t())
        x_recon = self.decode(z_mean) + self.batch_norm1(x) + self.batch_norm2(x_t).T
        return x_recon, z_mean, z_dropout, z_dispersion


def run_model(input_data, verbose=False, device=False):
    """Run model

    input_data: gene expression matrix
    params: hyperparameters
    clustering: whether to add batch normalized data
    """

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
    x_t, adj_t = get_data(input_data.T)

    x = x.to(device)
    adj = adj.to(device)
    x_t = x_t.to(device)
    adj_t = adj_t.to(device)

    params["input_dim"] = input_data.shape[1]
    params["hidden0"] = input_data.shape[0]

    model = VGAE(params).to(device)
    optimizer_name = "Adam"
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(),
        lr=params["lr"],
    )

    losses = []
    res = pd.DataFrame()

    if verbose:
        epochs = tqdm(range(params["epochs"]))
    else:
        epochs = range(params["epochs"])

    for epoch in epochs:
        x_recon, z_mean, z_dropout, z_dispersion = model(x, adj, x_t, adj_t)

        # Pass the ZINB outputs in their intended order:
        # mean, zero-inflation probability, dispersion.
        loss = compute_loss(
            x, x_recon, z_mean, z_dropout, z_dispersion, params["alpha"]
        ).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    pred = x_recon.cpu().detach().numpy()
    return pred
