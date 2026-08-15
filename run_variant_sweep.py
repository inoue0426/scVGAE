import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scVGAE
from run_all_clustering import (
    cluster_kmeans,
    historical_preprocess,
    load_dataset,
    set_seed,
)

# Fast hyperparameter screening. Final candidates should be re-evaluated with
# the historical paper clustering protocol in run_all_clustering.py.
VARIANTS = [
    ("baseline", 30, 50, 128),
    ("k15", 15, 50, 128),
    ("latent64", 30, 50, 64),
    ("k15_latent64", 15, 50, 64),
]


def save_results(rows, output_dir):
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "variant_results.csv", index=False)
    with open(output_dir / "variant_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    good = frame[frame["status"] == "ok"] if not frame.empty else frame
    if not good.empty:
        summary = (
            good.groupby("variant", as_index=True)[["ARI", "AMI"]]
            .mean()
            .sort_values("ARI", ascending=False)
        )
        summary.to_csv(output_dir / "variant_summary.csv")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast scVGAE variant sweep using KMeans only. Use the paper protocol "
            "only after selecting the best one or two variants."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["baron", "carey", "Fujii", "hcabm40k"],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="variant_sweep_kmeans")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset in args.datasets:
        print(f"\n===== {dataset} =====", flush=True)

        try:
            raw, source = load_dataset(args.data_root, dataset)
            processed, labels = historical_preprocess(raw)
        except Exception as exc:
            print(
                f"dataset load/preprocess failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": None,
                    "graph_k": None,
                    "pca_dim": None,
                    "latent_dim": None,
                    "ARI": None,
                    "AMI": None,
                    "best_ari_method": None,
                    "best_ami_method": None,
                    "elapsed_seconds": 0.0,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            )
            save_results(rows, output_dir)
            continue

        print(
            f"source={source} | {processed.shape[0]} cells x "
            f"{processed.shape[1]} genes | {len(np.unique(labels))} classes",
            flush=True,
        )

        for name, graph_k, pca_dim, latent_dim in VARIANTS:
            print(
                f"  [{name}] k={graph_k} pca={pca_dim} latent={latent_dim}",
                flush=True,
            )
            started = time.perf_counter()

            try:
                set_seed(args.seed)
                prediction = scVGAE.run_model(
                    processed,
                    verbose=False,
                    device=device,
                    graph_k=graph_k,
                    graph_pca_dim=pca_dim,
                    latent_dim=latent_dim,
                )

                if not np.isfinite(prediction).all():
                    raise RuntimeError("Prediction contains NaN or infinity")

                metrics = cluster_kmeans(prediction, labels, args.seed)
                elapsed = time.perf_counter() - started
                row = {
                    "dataset": dataset,
                    "variant": name,
                    "graph_k": graph_k,
                    "pca_dim": pca_dim,
                    "latent_dim": latent_dim,
                    "ARI": metrics["ARI"],
                    "AMI": metrics["AMI"],
                    "best_ari_method": metrics["best_ari_method"],
                    "best_ami_method": metrics["best_ami_method"],
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                }
                print(
                    f"    ARI={row['ARI']:.4f} AMI={row['AMI']:.4f} "
                    f"time={elapsed / 60:.1f} min",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                row = {
                    "dataset": dataset,
                    "variant": name,
                    "graph_k": graph_k,
                    "pca_dim": pca_dim,
                    "latent_dim": latent_dim,
                    "ARI": None,
                    "AMI": None,
                    "best_ari_method": None,
                    "best_ami_method": None,
                    "elapsed_seconds": elapsed,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
                print(f"    ERROR: {row['status']}", flush=True)

            rows.append(row)
            save_results(rows, output_dir)

            # Release per-variant prediction/model allocations before the next run.
            if "prediction" in locals():
                del prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    good = frame[frame["status"] == "ok"] if not frame.empty else frame
    print("\n===== Mean KMeans performance =====", flush=True)
    if good.empty:
        print("No successful variants.", flush=True)
    else:
        summary = (
            good.groupby("variant")[["ARI", "AMI"]]
            .mean()
            .sort_values("ARI", ascending=False)
        )
        print(summary.to_string(), flush=True)
        summary.to_csv(output_dir / "variant_summary.csv")

    print(f"\nResults written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
