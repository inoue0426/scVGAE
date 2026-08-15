import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss

import scVGAE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_once(input_data, loss_name, epochs, seed, device, beta):
    set_seed(seed)

    params = {
        "dropout1": 0.2,
        "dropout2": 0.4,
        "epochs": epochs,
        "hidden1": 128,
        "hidden2": 1024,
        "lr": 0.0001,
        "alpha": 0.05,
        "beta": beta,
        "input_dim": input_data.shape[1],
        "hidden0": input_data.shape[0],
        "latent_dim": input_data.shape[1],
    }

    x, adj = scVGAE.get_data(input_data)
    x_t, adj_t = scVGAE.get_data(input_data.T)
    x, adj = x.to(device), adj.to(device)
    x_t, adj_t = x_t.to(device), adj_t.to(device)

    model = scVGAE.VGAE(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    mse_loss_fn = MSELoss()
    zinb_loss_fn = (
        scVGAE.ZINBLoss if loss_name == "standard" else scVGAE.legacy_ZINBLoss
    )

    total_losses, zinb_losses, mse_losses, kl_losses = [], [], [], []
    start = time.perf_counter()

    model.train()
    for _ in range(epochs):
        outputs = model(x, adj, x_t, adj_t)
        x_recon, z_mean, z_dropout, z_dispersion, mu_z, logvar_z = outputs

        zinb_loss = zinb_loss_fn(x, z_mean, z_dispersion, z_dropout)
        mse_loss = mse_loss_fn(x_recon, x)
        kl_loss = scVGAE.KLDLoss(mu_z, logvar_z)
        total_loss = (
            params["alpha"] * zinb_loss
            + (1 - params["alpha"]) * mse_loss
            + params["beta"] * kl_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_losses.append(float(total_loss.detach().cpu()))
        zinb_losses.append(float(zinb_loss.detach().cpu()))
        mse_losses.append(float(mse_loss.detach().cpu()))
        kl_losses.append(float(kl_loss.detach().cpu()))

    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        outputs = model(x, adj, x_t, adj_t)
        pred, z_mean, z_dropout, z_dispersion, mu_z, logvar_z = outputs
        final_zinb = zinb_loss_fn(x, z_mean, z_dispersion, z_dropout)
        final_mse = mse_loss_fn(pred, x)
        final_kl = scVGAE.KLDLoss(mu_z, logvar_z)
        final_total = (
            params["alpha"] * final_zinb
            + (1 - params["alpha"]) * final_mse
            + params["beta"] * final_kl
        )

    return {
        "prediction": pred.detach().cpu().numpy(),
        "elapsed_seconds": elapsed,
        "final_total_loss": float(final_total.detach().cpu()),
        "final_zinb_loss": float(final_zinb.detach().cpu()),
        "final_mse_loss": float(final_mse.detach().cpu()),
        "final_kl_loss": float(final_kl.detach().cpu()),
        "total_loss_history": total_losses,
        "zinb_loss_history": zinb_losses,
        "mse_loss_history": mse_losses,
        "kl_loss_history": kl_losses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard and historical ZINB losses inside variational scVGAE."
    )
    parser.add_argument("--input", default="data/sample_data.csv.gz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--output-dir", default="zinb_comparison")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    data = pd.read_csv(args.input, index_col=0)
    if args.max_cells is not None:
        data = data.iloc[: args.max_cells].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for loss_name in ("legacy", "standard"):
        print(
            f"Running variational scVGAE with {loss_name} ZINB on "
            f"{data.shape[0]} cells x {data.shape[1]} genes..."
        )
        results[loss_name] = train_once(
            data, loss_name, args.epochs, args.seed, device, args.beta
        )

        np.save(
            output_dir / f"prediction_{loss_name}.npy",
            results[loss_name]["prediction"],
        )
        pd.DataFrame(
            {
                "epoch": np.arange(1, args.epochs + 1),
                "total_loss": results[loss_name]["total_loss_history"],
                "zinb_loss": results[loss_name]["zinb_loss_history"],
                "mse_loss": results[loss_name]["mse_loss_history"],
                "kl_loss": results[loss_name]["kl_loss_history"],
            }
        ).to_csv(output_dir / f"losses_{loss_name}.csv", index=False)

    diff = results["standard"]["prediction"] - results["legacy"]["prediction"]
    summary = {
        "input": args.input,
        "shape": list(data.shape),
        "epochs": args.epochs,
        "seed": args.seed,
        "beta": args.beta,
        "device": str(device),
        "legacy": {
            key: results["legacy"][key]
            for key in (
                "elapsed_seconds",
                "final_total_loss",
                "final_zinb_loss",
                "final_mse_loss",
                "final_kl_loss",
            )
        },
        "standard": {
            key: results["standard"][key]
            for key in (
                "elapsed_seconds",
                "final_total_loss",
                "final_zinb_loss",
                "final_mse_loss",
                "final_kl_loss",
            )
        },
        "prediction_difference": {
            "mean_absolute_difference": float(np.mean(np.abs(diff))),
            "root_mean_squared_difference": float(np.sqrt(np.mean(diff**2))),
            "max_absolute_difference": float(np.max(np.abs(diff))),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
