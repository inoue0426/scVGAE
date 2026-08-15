import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scVGAE
from run_all_clustering import (
    PAPER_DATASETS,
    cluster_kmeans,
    cluster_paper_protocol,
    historical_preprocess,
    load_dataset,
    set_seed,
)

ABLATIONS = ["full", "no_kl", "no_zinb", "no_recon", "deterministic"]


def save_results(records, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": records}, handle, indent=2)

    good = frame[frame.get("status", "") == "ok"].copy()
    if not good.empty:
        summary = good.groupby("ablation")[["ARI", "AMI"]].mean().reindex(ABLATIONS)
        summary.to_csv(output_dir / "ablation_summary.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Run component-wise ablations for the variational scVGAE model."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=PAPER_DATASETS)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=ABLATIONS,
        default=ABLATIONS,
        help="Ablations to run. Defaults to all five predefined variants.",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["kmeans", "paper"],
        default="paper",
        help="paper matches the manuscript evaluation; kmeans is faster for debugging.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--graph-k", type=int, default=30)
    parser.add_argument("--graph-pca-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="ablation_results_vgae")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run completed dataset/ablation pairs.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_results.csv"

    records = []
    completed = set()
    if results_path.exists() and not args.no_resume:
        previous = pd.read_csv(results_path)
        records = previous.to_dict("records")
        if {"dataset", "ablation", "status"}.issubset(previous.columns):
            ok = previous[previous["status"].astype(str) == "ok"]
            completed = set(zip(ok["dataset"].astype(str), ok["ablation"].astype(str)))
        print(f"Resuming: {len(completed)} successful dataset/ablation pairs found.")

    metadata = {
        "datasets": args.datasets,
        "ablations": args.ablations,
        "cluster_mode": args.cluster_mode,
        "latent_dim": args.latent_dim,
        "graph_k": args.graph_k,
        "graph_pca_dim": args.graph_pca_dim,
        "seed": args.seed,
        "device": str(device),
        "alpha": 0.05,
        "beta": 0.0001,
        "epochs": 100,
        "learning_rate": 0.0001,
        "definitions": {
            "full": "ZINB + MSE + KL with stochastic reparameterization",
            "no_kl": "ZINB + MSE with stochastic reparameterization, KL removed",
            "no_zinb": "MSE + KL, ZINB term removed",
            "no_recon": "ZINB + KL, MSE reconstruction term removed",
            "deterministic": "ZINB + MSE with z=mu and KL removed",
        },
    }

    for dataset_index, dataset in enumerate(args.datasets, start=1):
        print(
            f"\n[{dataset_index}/{len(args.datasets)}] {dataset}: loading", flush=True
        )
        try:
            raw, source = load_dataset(args.data_root, dataset)
            processed, labels = historical_preprocess(raw)
        except Exception as exc:
            print(f"  DATA ERROR: {type(exc).__name__}: {exc}", flush=True)
            for ablation in args.ablations:
                key = (dataset, ablation)
                if key in completed:
                    continue
                records = [
                    row
                    for row in records
                    if (row.get("dataset"), row.get("ablation")) != key
                ]
                records.append(
                    {
                        "dataset": dataset,
                        "ablation": ablation,
                        "ARI": None,
                        "AMI": None,
                        "cells": None,
                        "genes": None,
                        "classes": None,
                        "elapsed_seconds": 0.0,
                        "status": f"error: {type(exc).__name__}: {exc}",
                    }
                )
            save_results(records, output_dir, metadata)
            continue

        print(
            f"  source={source} | {processed.shape[0]} cells x {processed.shape[1]} genes "
            f"| classes={len(np.unique(labels))}",
            flush=True,
        )

        for ablation in args.ablations:
            key = (dataset, ablation)
            if key in completed:
                print(f"  [{ablation}] already complete, skipping", flush=True)
                continue

            print(f"  [{ablation}]", flush=True)
            started = time.perf_counter()
            try:
                set_seed(args.seed)
                prediction = scVGAE.run_model(
                    processed,
                    verbose=False,
                    device=device,
                    graph_k=args.graph_k,
                    graph_pca_dim=args.graph_pca_dim,
                    latent_dim=args.latent_dim,
                    ablation=ablation,
                )

                if not np.isfinite(prediction).all():
                    raise RuntimeError("Prediction contains NaN or infinity")

                if args.cluster_mode == "paper":
                    metrics = cluster_paper_protocol(prediction, labels, args.seed)
                else:
                    metrics = cluster_kmeans(prediction, labels, args.seed)

                elapsed = time.perf_counter() - started
                record = {
                    "dataset": dataset,
                    "ablation": ablation,
                    "ARI": metrics["ARI"],
                    "AMI": metrics["AMI"],
                    "best_ari_method": metrics["best_ari_method"],
                    "best_ami_method": metrics["best_ami_method"],
                    "cells": int(processed.shape[0]),
                    "genes": int(processed.shape[1]),
                    "classes": int(len(np.unique(labels))),
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                }
                print(
                    f"    ARI={record['ARI']:.4f} AMI={record['AMI']:.4f} "
                    f"time={elapsed / 60:.1f} min",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                record = {
                    "dataset": dataset,
                    "ablation": ablation,
                    "ARI": None,
                    "AMI": None,
                    "best_ari_method": None,
                    "best_ami_method": None,
                    "cells": int(processed.shape[0]),
                    "genes": int(processed.shape[1]),
                    "classes": int(len(np.unique(labels))),
                    "elapsed_seconds": elapsed,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
                print(f"    ERROR: {record['status']}", flush=True)

            records = [
                row
                for row in records
                if (row.get("dataset"), row.get("ablation")) != key
            ]
            records.append(record)
            save_results(records, output_dir, metadata)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(records)
    good = frame[frame["status"].astype(str) == "ok"]
    print("\n===== Mean performance =====", flush=True)
    if good.empty:
        print("No successful runs.", flush=True)
    else:
        summary = good.groupby("ablation")[["ARI", "AMI"]].mean().reindex(ABLATIONS)
        print(summary.to_string(), flush=True)
        summary.to_csv(output_dir / "ablation_summary.csv")
    print(f"\nResults written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
