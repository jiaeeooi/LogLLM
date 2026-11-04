from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

dataset_name = "BGL"
embedding = "projector" # "encoder" or "projector"

# -----------------------------------------------

DATA_DIR = Path(f"/content/drive/MyDrive/LogLLM/{dataset_name}")
PROBE_DIR = DATA_DIR / "lr_probe"

ENCODERS = ["author", "reproduced", "bert", "sbert", "bge", "qwen"]

RANDOM_SEED = 42

POSITIVE_LABEL = 1

def load_embeddings(encoder_name, split_name):
    file_path = (PROBE_DIR/encoder_name/embedding/f"{split_name}_embeddings.npz")

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
        raise ValueError(f"{encoder_name}/{split_name}: embeddings must be 2D, got {X.shape}")
    if y_raw.ndim != 1:
        raise ValueError(f"{encoder_name}/{split_name}: labels must be 1D, got {y_raw.shape}")
    if not (len(X) == len(y_raw) == len(content)):
        raise ValueError(
            f"{encoder_name}/{split_name}: row count mismatch: X={len(X)}, y={len(y_raw)}, content={len(content)}")

    y_numeric = pd.to_numeric(pd.Series(y_raw), errors="raise").to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(y_numeric)):
        raise ValueError(f"{encoder_name}/{split_name}: labels contain NaN or infinite values.")

    if not np.all(y_numeric == np.floor(y_numeric)):
        raise ValueError(f"{encoder_name}/{split_name}: labels must be integers.")

    y = y_numeric.astype(np.int64)

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{encoder_name}/{split_name}: expected binary labels 0 and 1, got {np.unique(y)}")

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{encoder_name}/{split_name}: embeddings contain NaN or infinite values.")

    return X, y, content

datasets = {}

for encoder_name in ENCODERS:
    print(f"Loading {encoder_name}...")

    X_train, y_train, train_content = load_embeddings(encoder_name, "train")
    X_test, y_test, test_content = load_embeddings(encoder_name, "test")

    if set(train_content) & set(test_content):
        raise ValueError(f"{encoder_name}: train/test template overlap detected.")

    # All encoders must use the same examples and label order
    if datasets:
        reference_name = ENCODERS[0]
        ref = datasets[reference_name]

        if not np.array_equal(train_content, ref["train_content"]):
            raise ValueError(f"{encoder_name}: training content/order differs from {reference_name}.")

        if not np.array_equal(test_content, ref["test_content"]):
            raise ValueError(f"{encoder_name}: test content/order differs from {reference_name}.")

        if not np.array_equal(y_train, ref["y_train"]):
            raise ValueError(f"{encoder_name}: training labels differ from {reference_name}.")

        if not np.array_equal(y_test, ref["y_test"]):
            raise ValueError(f"{encoder_name}: test labels differ from {reference_name}.")

    datasets[encoder_name] = {
        "X_train": X_train,
        "y_train": y_train,
        "train_content": train_content,
        "X_test": X_test,
        "y_test": y_test,
        "test_content": test_content,
    }

    print(
        f"Train: {X_train.shape}, "
        f"Test: {X_test.shape}"
    )

print("\nAll embedding files loaded and validated.")

model_outputs = {}
results = []

for encoder_name in ENCODERS:
    data = datasets[encoder_name]

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("\n" + "=" * 65)
    print(f"Encoder: {encoder_name}")
    print("=" * 65)

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples:     {len(y_test)}")
    print(f"Embedding dim:    {X_train.shape[1]}")

    # Use the same pipeline and hyperparameters for every encoder.
    # The scaler is fitted on TRAINING data only.
    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
            C=1.0,
        )),
    ])

    # Fit on training embeddings only
    probe.fit(X_train, y_train)

    # Evaluate on held-out test embeddings
    y_pred = probe.predict(X_test)
    y_score = probe.predict_proba(X_test)[:, 1]

    model_outputs[encoder_name] = {
        "y_pred": y_pred,
        "y_score": y_score,
    }

    # Metrics for anomalous class = 1
    metrics = {
        "encoder": encoder_name,
        "embedding_dim": X_train.shape[1],
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "ROC_AUC": roc_auc_score(y_test, y_score),
        "PR_AUC_AP": average_precision_score(y_test, y_score),
        "F1": f1_score(
            y_test, y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_test, y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test, y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "Accuracy": accuracy_score(y_test, y_pred),
    }

    results.append(metrics)

    for metric_name, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{metric_name:16s}: {value:.4f}")

    print(
        f"Predicted anomalies: "
        f"{int(np.sum(y_pred == POSITIVE_LABEL))} / {len(y_pred)}"
    )

# COMPARISON TABLE
results_df = pd.DataFrame(results)

metric_columns = ["ROC_AUC", "PR_AUC_AP", "F1", "Precision", "Recall", "Accuracy"]

# Sort by ROC-AUC for convenient comparison
results_df = results_df.sort_values(
    by="ROC_AUC", ascending=False
).reset_index(drop=True)

print("\n\n" + "=" * 100)
print(f"LOGISTIC REGRESSION PROBE — {embedding.upper()} EMBEDDING COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            col: "{:.4f}".format
            for col in metric_columns
        },
    )
)

# Save the full-precision results
output_csv = PROBE_DIR / f"lrprobe_results_{embedding}.csv"
results_df.to_csv(output_csv, index=False)

print(f"\nResults saved to:\n{output_csv}")

# -----------------------------------------------
# STRATIFIED PAIRED BOOTSTRAP
# -----------------------------------------------

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 123

rng = np.random.default_rng(BOOTSTRAP_SEED)

# Separate test indices by class
normal_indices = np.where(y_test == 0)[0]
anomaly_indices = np.where(y_test == 1)[0]

print("\nBootstrap test set:")
print(f"Normal templates:   {len(normal_indices)}")
print(f"Anomalous templates: {len(anomaly_indices)}")

# Metrics to calculate
BOOTSTRAP_METRICS = ["ROC_AUC", "AP", "F1"]

# Store bootstrap results for every encoder
bootstrap_results = {
    encoder: {
        metric: np.zeros(N_BOOTSTRAP)
        for metric in BOOTSTRAP_METRICS
    }
    for encoder in ENCODERS
}

# Perform bootstrap sampling
for i in range(N_BOOTSTRAP):
    # Stratified sampling: preserve the original class counts
    sampled_normal = rng.choice(normal_indices, size=len(normal_indices), replace=True)
    sampled_anomaly = rng.choice(anomaly_indices, size=len(anomaly_indices), replace=True)
    sampled_indices = np.concatenate([sampled_normal, sampled_anomaly])
    y_boot = y_test[sampled_indices]

    for encoder_name in ENCODERS:
        y_pred_boot = model_outputs[encoder_name]["y_pred"][sampled_indices]
        y_score_boot = model_outputs[encoder_name]["y_score"][sampled_indices]

        bootstrap_results[encoder_name]["ROC_AUC"][i] = roc_auc_score(y_boot, y_score_boot)
        bootstrap_results[encoder_name]["AP"][i] = average_precision_score(y_boot, y_score_boot)
        bootstrap_results[encoder_name]["F1"][i] = f1_score(y_boot, y_pred_boot, pos_label=POSITIVE_LABEL, zero_division=0)

print(f"\nCompleted {N_BOOTSTRAP} bootstrap iterations.")


# -----------------------------------------------
# 95% CONFIDENCE INTERVALS FOR EACH ENCODER
# -----------------------------------------------

ci_results = []
for encoder_name in ENCODERS:
    for metric in BOOTSTRAP_METRICS:
        values = bootstrap_results[encoder_name][metric]
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)

        # Original test-set metric
        original_value = results_df.loc[results_df["encoder"] == encoder_name, "PR_AUC_AP" if metric == "AP" else metric].iloc[0]

        ci_results.append({
            "encoder": encoder_name,
            "metric": metric,
            "original_value": original_value,
            "CI_lower": lower,
            "CI_upper": upper,
        })

ci_df = pd.DataFrame(ci_results)

print("\n" + "=" * 90)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 90)

print(
    ci_df.to_string(
        index=False,
        formatters={
            "original_value": "{:.4f}".format,
            "CI_lower": "{:.4f}".format,
            "CI_upper": "{:.4f}".format,
        },
    )
)

# Save confidence intervals
ci_output = PROBE_DIR / f"lrprobe_bootstrap_ci_{embedding}.csv"
ci_df.to_csv(ci_output, index=False)

print(f"\nConfidence intervals saved to:\n{ci_output}")

# -----------------------------------------------
# PAIRED BOOTSTRAP: DIFFERENCES BETWEEN ENCODERS
# -----------------------------------------------

# Comparisons of interest
COMPARISONS = [
    ("bge", "sbert"),
    ("reproduced", "author"),
    ("sbert", "bert"),
    ("bge", "bert"),
]

difference_results = []

for encoder_a, encoder_b in COMPARISONS:
    for metric in BOOTSTRAP_METRICS:
        # Difference in every paired bootstrap iteration
        differences = bootstrap_results[encoder_a][metric] - bootstrap_results[encoder_b][metric]
        lower = np.percentile(differences, 2.5)
        upper = np.percentile(differences, 97.5)

        # Difference in the original test-set results
        column = "PR_AUC_AP" if metric == "AP" else metric

        value_a = results_df.loc[results_df["encoder"] == encoder_a, column].iloc[0]
        value_b = results_df.loc[results_df["encoder"] == encoder_b, column].iloc[0]

        difference_results.append({
            "comparison": f"{encoder_a} - {encoder_b}",
            "metric": metric,
            "original_difference": value_a - value_b,
            "CI_lower": lower,
            "CI_upper": upper,
        })

difference_df = pd.DataFrame(difference_results)

print("\n" + "=" * 90)
print("PAIRED BOOTSTRAP: DIFFERENCES BETWEEN ENCODERS")
print("=" * 90)

print(
    difference_df.to_string(
        index=False,
        formatters={
            "original_difference": "{:+.4f}".format,
            "CI_lower": "{:+.4f}".format,
            "CI_upper": "{:+.4f}".format,
        },
    )
)

# Save paired differences
difference_output = PROBE_DIR / f"lrprobe_bootstrap_differences_{embedding}.csv"
difference_df.to_csv(difference_output, index=False)

print(f"\nPaired differences saved to:\n{difference_output}")