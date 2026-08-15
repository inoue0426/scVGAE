import argparse
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import scVGAE

PAPER_DATASETS = [
    "baron",
    "brosens",
    "carey",
    "cbmc",
    "chang",
    "Fujii",
    "hcabm40k",
    "hrvatin",
    "jakel",
    "jiang",
    "manno",
    "mingyao",
    "pbmc3k",
    "Xu",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_root, dataset):
    """Load historical paper data from either an extracted directory or zip file."""
    data_root = Path(data_root)
    direct = data_root / dataset / "data.csv.gz"
    if direct.exists():
        return pd.read_csv(direct, index_col=0), str(direct)

    archive = data_root / f"{dataset}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            members = [
                name
                for name in zf.namelist()
                if name.endswith("data.csv.gz") and not name.startswith("__MACOSX/")
            ]
            if len(members) != 1:
                raise RuntimeError(
                    f"Expected one data.csv.gz in {archive}, found {len(members)}: {members}"
                )
            with zf.open(members[0]) as handle:
                return pd.read_csv(handle, index_col=0), f"{archive}:{members[0]}"

    raise FileNotFoundError(
        f"Could not find {direct} or {archive}. "
        "Point --data-root to the data directory from inoue0426/scVGAE-paper."
    )


def historical_preprocess(df):
    """Reproduce preprocessing used by the historical clustering notebooks.

    1. Retain genes nonzero in >5% of cells.
    2. Retain cells nonzero in >5% of genes.
    3. log1p transform.
    4. Library-size normalize each cell to total 10,000.
    5. Square-root transform.

    Cell-type labels are taken from the DataFrame index.
    """
    nonzero = np.sign(df.to_numpy())
    col_mask = nonzero.sum(axis=0) > int(df.shape[0] * 0.05)
    row_mask = nonzero.sum(axis=1) > int(df.shape[1] * 0.05)

    filtered = df.loc[row_mask, col_mask].astype(np.float64)
    labels = filtered.index.to_numpy(copy=True)

    x = np.log1p(filtered.to_numpy())
    library_size = x.sum(axis=1, keepdims=True)
    library_size[library_size == 0] = 1.0
    x = x / library_size * 10000.0
    x = np.sqrt(x)

    processed = pd.DataFrame(x, index=filtered.index, columns=filtered.columns)
    return processed, labels


def cluster_kmeans(prediction, labels, seed):
    n_clusters = len(np.unique(labels))
    clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        prediction
    )
    return {
        "ARI": float(adjusted_rand_score(labels, clusters)),
        "AMI": float(adjusted_mutual_info_score(labels, clusters)),
        "best_ari_method": "kmeans",
        "best_ami_method": "kmeans",
    }


def cluster_paper_protocol(prediction, labels, seed):
    """Reproduce the historical paper's oracle-style clustering comparison.

    The historical notebook evaluated KMeans and SpectralClustering with cosine,
    linear, and polynomial affinities, then reported the maximum ARI and maximum
    AMI independently across those clusterers. This mode is for comparison with
    the previously reported table; kmeans mode is preferable for a single fixed
    evaluation protocol.
    """
    n_clusters = len(np.unique(labels))
    candidates = []

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        prediction
    )
    candidates.append(
        (
            "kmeans",
            float(adjusted_rand_score(labels, km)),
            float(adjusted_mutual_info_score(labels, km)),
        )
    )

    for affinity in ("cosine", "linear", "poly"):
        try:
            clusters = SpectralClustering(
                n_clusters=n_clusters,
                random_state=seed,
                affinity=affinity,
            ).fit_predict(prediction)
            candidates.append(
                (
                    f"spectral_{affinity}",
                    float(adjusted_rand_score(labels, clusters)),
                    float(adjusted_mutual_info_score(labels, clusters)),
                )
            )
        except Exception as exc:
            print(f"  spectral_{affinity} failed: {type(exc).__name__}: {exc}")

    best_ari = max(candidates, key=lambda item: item[1])
    best_ami = max(candidates, key=lambda item: item[2])
    return {
        "ARI": best_ari[1],
        "AMI": best_ami[2],
        "best_ari_method": best_ari[0],
        "best_ami_method": best_ami[0],
        "all_clusterers": [
            {"method": name, "ARI": ari, "AMI": ami} for name, ari, ami in candidates
        ],
    }


def save_results(records, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "ari_ami_results.csv", index=False)
    with open(output_dir / "ari_ami_results.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": records}, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run variational scVGAE on the 14 paper datasets and report ARI/AMI."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to scVGAE-paper/data. Extracted dataset dirs or original zip files both work.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=PAPER_DATASETS,
        help="Dataset names to run. Defaults to the 14 datasets in the paper.",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["kmeans", "paper"],
        default="kmeans",
        help="kmeans is fixed and much faster; paper reproduces the historical max-over-clusterers protocol.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        default="all_dataset_results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save each imputed matrix as .npy. Disabled by default because files can be large.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run datasets even when they already exist in the output CSV.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    if args.save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "ari_ami_results.csv"
    records = []
    completed = set()
    if results_path.exists() and not args.no_resume:
        previous = pd.read_csv(results_path)
        if "dataset" in previous.columns:
            records = previous.to_dict("records")
            completed = set(previous["dataset"].astype(str))
            print(f"Resuming: {len(completed)} completed datasets found.")

    metadata = {
        "datasets": args.datasets,
        "cluster_mode": args.cluster_mode,
        "seed": args.seed,
        "device": str(device),
        "model": "variational scVGAE with standard ZINB and KL",
        "alpha": 0.05,
        "beta": 0.0001,
        "epochs": 100,
        "learning_rate": 0.0001,
    }

    for index, dataset in enumerate(args.datasets, start=1):
        if dataset in completed:
            print(
                f"[{index}/{len(args.datasets)}] {dataset}: already complete, skipping"
            )
            continue

        print(f"[{index}/{len(args.datasets)}] {dataset}: loading")
        started = time.perf_counter()
        try:
            raw, source = load_dataset(args.data_root, dataset)
            processed, labels = historical_preprocess(raw)
            print(
                f"  source={source} | processed={processed.shape[0]} cells x "
                f"{processed.shape[1]} genes | classes={len(np.unique(labels))}"
            )

            set_seed(args.seed)
            prediction = scVGAE.run_model(processed, verbose=False, device=device)

            if not np.isfinite(prediction).all():
                raise RuntimeError("Prediction contains NaN or infinity")

            if args.cluster_mode == "paper":
                metrics = cluster_paper_protocol(prediction, labels, args.seed)
            else:
                metrics = cluster_kmeans(prediction, labels, args.seed)

            elapsed = time.perf_counter() - started
            record = {
                "dataset": dataset,
                "cells": int(processed.shape[0]),
                "genes": int(processed.shape[1]),
                "classes": int(len(np.unique(labels))),
                "ARI": metrics["ARI"],
                "AMI": metrics["AMI"],
                "best_ari_method": metrics["best_ari_method"],
                "best_ami_method": metrics["best_ami_method"],
                "elapsed_seconds": elapsed,
                "status": "ok",
            }
            if "all_clusterers" in metrics:
                with open(
                    output_dir / f"{dataset}_clusterers.json", "w", encoding="utf-8"
                ) as handle:
                    json.dump(metrics["all_clusterers"], handle, indent=2)

            if args.save_predictions:
                np.save(predictions_dir / f"{dataset}.npy", prediction)

            print(
                f"  ARI={record['ARI']:.4f} AMI={record['AMI']:.4f} "
                f"time={elapsed / 60:.1f} min"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            record = {
                "dataset": dataset,
                "cells": None,
                "genes": None,
                "classes": None,
                "ARI": None,
                "AMI": None,
                "best_ari_method": None,
                "best_ami_method": None,
                "elapsed_seconds": elapsed,
                "status": f"error: {type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {record['status']}")

        records = [row for row in records if row.get("dataset") != dataset]
        records.append(record)
        save_results(records, output_dir, metadata)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nFinal results:")
    frame = pd.DataFrame(records)
    columns = [
        col
        for col in ["dataset", "ARI", "AMI", "cells", "genes", "classes", "status"]
        if col in frame.columns
    ]
    print(frame[columns].to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
