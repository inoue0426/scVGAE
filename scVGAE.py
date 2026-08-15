import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import pairwise_kernels
from torch.nn import BatchNorm1d, Linear, Module, MSELoss
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv, GraphNorm
from tqdm import tqdm


def get_topX(X):
    return X * np.array(X > np.percentile(X, 85), dtype=int)


def get_adj(x):
    row, col = x.nonzero()
    return torch.tensor(np.vstack((row, col)), dtype=torch.long)


def transpose_adj(edge_index):
    return edge_index.flip(0)


def get_data(X, metric="linear"):
    dist = pairwise_kernels(X, metric=metric)
    dist_x = get_topX(dist)
    return torch.tensor(X.values, dtype=torch.float), get_adj(dist_x)


def ZINBLoss(y_true, y_pred, theta, pi, eps=1e-10):
    """Standard zero-inflated negative-binomial negative log-likelihood."""
    y_pred = torch.clamp(y_pred, min=eps)
    theta = torch.clamp(theta, min=eps)
    pi = torch.clamp(pi, min=eps, max=1 - eps)

    log_theta_mu = torch.log(theta + y_pred + eps)
    nb_log_prob = (
        torch.lgamma(y_true + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y_true + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + y_true * (torch.log(y_pred + eps) - log_theta_mu)
    )
    nb_zero_log_prob = theta * (torch.log(theta + eps) - log_theta_mu)
    zero_log_prob = torch.logaddexp(torch.log(pi), torch.log1p(-pi) + nb_zero_log_prob)
    nonzero_log_prob = torch.log1p(-pi) + nb_log_prob
    log_prob = torch.where(y_true < eps, zero_log_prob, nonzero_log_prob)
    return -torch.sum(log_prob)


def legacy_ZINBLoss(y_true, y_pred, theta, pi, eps=1e-10):
    """Historical scVGAE ZINB expression retained only for comparison."""
    nb_terms = (
        -torch.lgamma(y_true + theta)
        + torch.lgamma(y_true + 1)
        + torch.lgamma(theta)
        - theta * torch.log(theta + eps)
        + theta * torch.log(theta + y_pred + eps)
        - y_true * torch.log(y_pred + theta + eps)
        + y_true * torch.log(y_pred + eps)
    )
    result = -torch.sum(
        torch.log(pi + (1 - pi) * torch.pow(1 + y_pred / theta, -theta))
        * (y_true < eps).float()
        + (1 - (y_true < eps).float()) * nb_terms
    )
    return torch.round(result, decimals=3)


def KLDLoss(mu_z, logvar_z):
    """KL[q(z|X,A) || N(0,I)], averaged across cells."""
    kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)
    return torch.mean(kl)


def compute_loss(
    x_original,
    x_recon,
    z_mean,
    z_dropout,
    z_dispersion,
    mu_z,
    logvar_z,
    alpha,
    beta,
):
    zinb_loss = ZINBLoss(x_original, z_mean, z_dispersion, z_dropout)
    mse_loss = MSELoss()(x_recon, x_original)
    kl_loss = KLDLoss(mu_z, logvar_z)
    reconstruction_loss = alpha * zinb_loss + (1 - alpha) * mse_loss
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, zinb_loss, mse_loss, kl_loss


class VGAE(Module):
    """Variational graph autoencoder with ZINB observation heads."""

    def __init__(self, params):
        super(VGAE, self).__init__()
        self.dropout1 = nn.Dropout(params["dropout1"])
        self.dropout2 = nn.Dropout(params["dropout2"])

        input_dim = params["input_dim"]
        hidden1 = params["hidden1"]
        hidden2 = params["hidden2"]
        latent_dim = params.get("latent_dim", input_dim)

        self.gcn1 = GCNConv(input_dim, hidden1)
        self.gn1 = GraphNorm(hidden1)
        self.gcn_mu = GCNConv(hidden1, latent_dim)
        self.gcn_logvar = GCNConv(hidden1, latent_dim)

        self.gcn_zinb_mean = GCNConv(latent_dim, input_dim)
        self.gcn_zinb_dropout = GCNConv(latent_dim, input_dim)
        self.gcn_zinb_dispersion = GCNConv(latent_dim, input_dim)

        self.fc1 = Linear(latent_dim, hidden2)
        self.bn2 = BatchNorm1d(hidden2)
        self.fc2 = Linear(hidden2, input_dim)

        self.batch_norm1 = BatchNorm1d(input_dim)
        self.batch_norm2 = BatchNorm1d(params["hidden0"])

    def encode(self, x, edge_index):
        h = relu(self.gn1(self.gcn1(x, edge_index)))
        h = self.dropout1(h)
        mu_z = self.gcn_mu(h, edge_index)
        logvar_z = torch.clamp(self.gcn_logvar(h, edge_index), min=-10.0, max=10.0)
        return mu_z, logvar_z

    def reparameterize(self, mu_z, logvar_z):
        if self.training:
            std = torch.exp(0.5 * logvar_z)
            return mu_z + torch.randn_like(std) * std
        return mu_z

    def decode_zinb(self, z, edge_index):
        z_mean = torch.exp(torch.clamp(self.gcn_zinb_mean(z, edge_index), max=15.0))
        z_dropout = torch.sigmoid(self.gcn_zinb_dropout(z, edge_index))
        z_dispersion = torch.exp(
            torch.clamp(self.gcn_zinb_dispersion(z, edge_index), max=15.0)
        )
        return z_mean, z_dropout, z_dispersion

    def decode_expression(self, z):
        h = relu(self.bn2(self.fc1(z)))
        h = self.dropout2(h)
        return relu(self.fc2(h))

    def forward(self, x, adj, x_t, adj_t):
        edge_index = transpose_adj(adj)
        mu_z, logvar_z = self.encode(x, edge_index)
        z = self.reparameterize(mu_z, logvar_z)
        z_mean, z_dropout, z_dispersion = self.decode_zinb(z, edge_index)
        x_recon = (
            self.decode_expression(z) + self.batch_norm1(x) + self.batch_norm2(x_t).T
        )
        return x_recon, z_mean, z_dropout, z_dispersion, mu_z, logvar_z


def run_model(input_data, verbose=False, device=False):
    """Run scVGAE using the variational graph autoencoder objective."""
    params = {
        "dropout1": 0.2,
        "dropout2": 0.4,
        "epochs": 100,
        "hidden1": 128,
        "hidden2": 1024,
        "lr": 0.0001,
        "alpha": 0.05,
        "beta": 0.0001,
    }

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, adj = get_data(input_data)
    x_t, adj_t = get_data(input_data.T)
    x, adj = x.to(device), adj.to(device)
    x_t, adj_t = x_t.to(device), adj_t.to(device)

    params["input_dim"] = input_data.shape[1]
    params["hidden0"] = input_data.shape[0]
    params["latent_dim"] = input_data.shape[1]

    model = VGAE(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    epochs = tqdm(range(params["epochs"])) if verbose else range(params["epochs"])

    for _ in epochs:
        outputs = model(x, adj, x_t, adj_t)
        x_recon, z_mean, z_dropout, z_dispersion, mu_z, logvar_z = outputs
        loss, _, _, _ = compute_loss(
            x,
            x_recon,
            z_mean,
            z_dropout,
            z_dispersion,
            mu_z,
            logvar_z,
            params["alpha"],
            params["beta"],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_recon, *_ = model(x, adj, x_t, adj_t)
    return x_recon.cpu().numpy()
