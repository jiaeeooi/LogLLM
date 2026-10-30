from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import umap

DATASETS = ["BGL", "Liberty"]
ENCODERS = ["author", "reproduced", "bert", "sbert", "bge", "qwen"]
EMBEDDINGS = ["encoder", "projector"]

RANDOM_SEED = 42
MAX_COSINE_PAIRS = 100000
SILHOUETTE_SAMPLE_SIZE = 5000
# UMAP settings
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

def load_embeddings(dataset_name, encoder_name, embedding):
    data_dir = Path(f"/content/drive/MyDrive/LogLLM/{dataset_name}/representation_analysis")
    file_path = (data_dir / encoder_name / embedding / "test_embeddings.npz")

    if not file_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {file_path}")

    with np.load(file_path, allow_pickle=True) as data:
        required_keys = {"embeddings", "labels", "content"}
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            raise ValueError(f"{file_path} is missing keys: {missing_keys}")

        X = np.asarray(data["embeddings"], dtype=np.float64)
        y_raw = np.asarray(data["labels"])
        content = np.asarray(data["content"]).astype(str)

    if X.ndim != 2:
        raise ValueError(f"{encoder_name}/{embedding}: embeddings must be 2D, got {X.shape}")
    if y_raw.ndim != 1:
        raise ValueError(f"{encoder_name}/{embedding}: labels must be 1D, got {y_raw.shape}")
    if not (len(X) == len(y_raw) == len(content)):
        raise ValueError(f"{encoder_name}/{embedding}: row count mismatch: X={len(X)}, y={len(y_raw)}, content={len(content)}")

    y_numeric = pd.to_numeric(pd.Series(y_raw), errors="raise").to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(y_numeric)):
        raise ValueError(f"{encoder_name}/{embedding}: labels contain NaN or infinite values.")
    if not np.all(y_numeric == np.floor(y_numeric)):
        raise ValueError(f"{encoder_name}/{embedding}: labels must be integers.")

    y = y_numeric.astype(np.int64)

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{encoder_name}/{embedding}: expected binary labels 0 and 1.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{encoder_name}/{embedding}: embeddings contain NaN or infinite values.")
    if len(np.unique(content)) != len(content):
        raise ValueError(
            f"{encoder_name}/{embedding}: duplicate log messages detected."
        )

    return X, y, content

def validate_embeddings(dataset_name, datasets):
    reference = None
    for embedding in EMBEDDINGS:
        for encoder_name in ENCODERS:
            print(f"Loading {encoder_name} - {embedding}...")
            X, y, content = load_embeddings(dataset_name, encoder_name, embedding)
            if reference is None:
                reference = {
                    "content": content,
                    "labels": y
                }
            else:
                if not np.array_equal(content, reference["content"]):
                    raise ValueError(f"{encoder_name}/{embedding}: content/order differs from the reference.")
                if not np.array_equal(y, reference["labels"]):
                    raise ValueError(f"{encoder_name}/{embedding}: labels differ from the reference.")

            datasets[(encoder_name, embedding)] = {
                "X": X,
                "y": y,
                "content": content
            }

            print(f"Shape: {X.shape}, Normal: {(y == 0).sum()}, Anomalous: {(y == 1).sum()}")
    print("\nAll embedding files loaded and validated.")


# ============================================================
# 4. Embedding norm analysis
# ============================================================

def calculate_norms(X, y):

    norms = np.linalg.norm(X, axis=1)

    results = {}

    for label, class_name in [(0, "normal"), (1, "anomalous")]:

        class_norms = norms[y == label]

        results[f"{class_name}_norm_mean"] = np.mean(class_norms)
        results[f"{class_name}_norm_std"] = np.std(class_norms)
        results[f"{class_name}_norm_median"] = np.median(class_norms)
        results[f"{class_name}_norm_min"] = np.min(class_norms)
        results[f"{class_name}_norm_max"] = np.max(class_norms)

    results["overall_norm_mean"] = np.mean(norms)
    results["overall_norm_std"] = np.std(norms)
    results["overall_norm_median"] = np.median(norms)

    return norms, results


# ============================================================
# 5. Cosine similarity analysis
# ============================================================

def sample_within_class_cosine(X_normalized, indices, max_pairs, rng):

    n_samples = len(indices)

    if n_samples < 2:
        return np.array([])

    n_pairs = min(
        max_pairs,
        n_samples * (n_samples - 1)
    )

    # Sample two different indices from the same class
    first_idx = rng.integers(0, n_samples, size=n_pairs)

    second_idx = rng.integers(0, n_samples - 1, size=n_pairs)

    second_idx += (second_idx >= first_idx)

    first = X_normalized[indices[first_idx]]
    second = X_normalized[indices[second_idx]]

    similarities = np.sum(first * second, axis=1)

    return similarities


def sample_between_class_cosine(
    X_normalized,
    indices_a,
    indices_b,
    max_pairs,
    rng
):

    if len(indices_a) == 0 or len(indices_b) == 0:
        return np.array([])

    n_pairs = min(
        max_pairs,
        len(indices_a) * len(indices_b)
    )

    first_idx = rng.choice(
        indices_a,
        size=n_pairs,
        replace=True
    )

    second_idx = rng.choice(
        indices_b,
        size=n_pairs,
        replace=True
    )

    similarities = np.sum(
        X_normalized[first_idx] * X_normalized[second_idx],
        axis=1
    )

    return similarities


def cosine_summary(similarities, name):

    if len(similarities) == 0:
        return {
            f"{name}_cosine_mean": np.nan,
            f"{name}_cosine_std": np.nan,
            f"{name}_cosine_median": np.nan
        }

    return {
        f"{name}_cosine_mean": np.mean(similarities),
        f"{name}_cosine_std": np.std(similarities),
        f"{name}_cosine_median": np.median(similarities)
    }


def calculate_cosine_similarity(X, y):

    # Normalize each embedding to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    X_normalized = X / np.maximum(norms, 1e-12)

    normal_indices = np.where(y == 0)[0]
    anomalous_indices = np.where(y == 1)[0]

    rng = np.random.default_rng(RANDOM_SEED)

    # Normal-normal similarity
    normal_normal = sample_within_class_cosine(
        X_normalized,
        normal_indices,
        MAX_COSINE_PAIRS,
        rng
    )

    # Anomalous-anomalous similarity
    anomalous_anomalous = sample_within_class_cosine(
        X_normalized,
        anomalous_indices,
        MAX_COSINE_PAIRS,
        rng
    )

    # Normal-anomalous similarity
    normal_anomalous = sample_between_class_cosine(
        X_normalized,
        normal_indices,
        anomalous_indices,
        MAX_COSINE_PAIRS,
        rng
    )

    results = {}

    results.update(
        cosine_summary(normal_normal, "normal_normal")
    )

    results.update(
        cosine_summary(anomalous_anomalous, "anomalous_anomalous")
    )

    results.update(
        cosine_summary(normal_anomalous, "normal_anomalous")
    )

    return results


# ============================================================
# 6. Silhouette score
# ============================================================

def calculate_silhouette(X, y):

    unique_labels, counts = np.unique(y, return_counts=True)

    if len(unique_labels) < 2 or np.min(counts) < 2:
        return np.nan

    sample_size = min(
        SILHOUETTE_SAMPLE_SIZE,
        len(X)
    )

    score = silhouette_score(
        X,
        y,
        metric="cosine",
        sample_size=sample_size if sample_size < len(X) else None,
        random_state=RANDOM_SEED
    )

    return score


# ============================================================
# 7. UMAP visualization
# ============================================================

def generate_umap(
    X,
    y,
    content,
    dataset_name,
    encoder_name,
    embedding,
    output_dir
):

    n_samples = len(X)

    if n_samples < 4:
        raise ValueError(
            f"Not enough samples for UMAP: {n_samples}"
        )

    n_neighbors = min(
        UMAP_NEIGHBORS,
        n_samples - 1
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED
    )

    X_umap = reducer.fit_transform(X)

    # Save UMAP coordinates
    umap_df = pd.DataFrame({
        "content": content,
        "label": y,
        "UMAP_1": X_umap[:, 0],
        "UMAP_2": X_umap[:, 1]
    })

    umap_csv = output_dir / (
        f"umap_{encoder_name}_{embedding}.csv"
    )

    umap_df.to_csv(umap_csv, index=False)

    # Plot
    plt.figure(figsize=(8, 6))

    normal_mask = y == 0
    anomalous_mask = y == 1

    plt.scatter(
        X_umap[normal_mask, 0],
        X_umap[normal_mask, 1],
        label="Normal",
        alpha=0.6,
        s=15
    )

    plt.scatter(
        X_umap[anomalous_mask, 0],
        X_umap[anomalous_mask, 1],
        label="Anomalous",
        alpha=0.7,
        s=20
    )

    plt.title(
        f"{dataset_name} - {encoder_name} - {embedding}"
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / (
        f"umap_{encoder_name}_{embedding}.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return X_umap


# ============================================================
# 8. Run representation analysis
# ============================================================

def run_analysis(dataset_name, datasets):

    results = []

    output_dir = Path(
        f"/content/drive/MyDrive/LogLLM/{dataset_name}/representation_analysis"
    )

    for embedding in EMBEDDINGS:

        for encoder_name in ENCODERS:

            print("\n" + "=" * 70)
            print(f"Dataset: {dataset_name}")
            print(f"Encoder: {encoder_name}")
            print(f"Embedding: {embedding}")
            print("=" * 70)

            data = datasets[(encoder_name, embedding)]

            X = data["X"]
            y = data["y"]
            content = data["content"]

            print(f"Samples: {len(X)}")
            print(f"Embedding dimension: {X.shape[1]}")

            # ----------------------------------------
            # Embedding norm
            # ----------------------------------------

            norms, norm_results = calculate_norms(X, y)

            # ----------------------------------------
            # Cosine similarity
            # ----------------------------------------

            cosine_results = calculate_cosine_similarity(X, y)

            # ----------------------------------------
            # Silhouette score
            # ----------------------------------------

            silhouette = calculate_silhouette(X, y)

            print(f"Silhouette score: {silhouette:.4f}")

            # ----------------------------------------
            # UMAP
            # ----------------------------------------

            X_umap = generate_umap(
                X,
                y,
                content,
                dataset_name,
                encoder_name,
                embedding,
                output_dir
            )

            # ----------------------------------------
            # Store results
            # ----------------------------------------

            metrics = {
                "dataset": dataset_name,
                "encoder": encoder_name,
                "embedding": embedding,
                "embedding_dim": X.shape[1],
                "num_samples": len(X),
                "num_normal": int(np.sum(y == 0)),
                "num_anomalous": int(np.sum(y == 1)),
                "silhouette_cosine": silhouette
            }

            metrics.update(norm_results)
            metrics.update(cosine_results)

            results.append(metrics)

    # ----------------------------------------
    # Save comparison table
    # ----------------------------------------

    results_df = pd.DataFrame(results)

    output_csv = output_dir / "representation_analysis_results.csv"

    results_df.to_csv(output_csv, index=False)

    print("\n\n" + "=" * 100)
    print(f"REPRESENTATION ANALYSIS — {dataset_name}")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(f"\nResults saved to:\n{output_csv}")

    return results_df


# ============================================================
# 9. Main
# ============================================================

def main():

    all_results = []

    for dataset_name in DATASETS:

        print("\n" + "#" * 80)
        print(f"PROCESSING DATASET: {dataset_name}")
        print("#" * 80)

        datasets = {}

        # Load and validate all 12 embeddings for this dataset
        validate_embeddings(
            dataset_name,
            datasets
        )

        # Run analysis
        results_df = run_analysis(
            dataset_name,
            datasets
        )

        all_results.append(results_df)

        del datasets

    # Combine BGL and Liberty results
    combined_results = pd.concat(
        all_results,
        ignore_index=True
    )

    combined_output = Path(
        "/content/drive/MyDrive/LogLLM/representation_analysis_all_results.csv"
    )

    combined_results.to_csv(
        combined_output,
        index=False
    )

    print("\nAll representation analyses completed.")
    print(f"Combined results saved to:\n{combined_output}")


if __name__ == "__main__":
    main()