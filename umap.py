import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

UMAP_OUTPUT = "/content/drive/MyDrive/LogLLM/results/embeddings_BGL/umap_author.csv"

# Paths
EMBEDDINGS_PATH = "/content/drive/MyDrive/LogLLM/results/embeddings_BGL/embeddings_author.npy"
METADATA_PATH = "/content/drive/MyDrive/LogLLM/results/embeddings_BGL/embedding_metadata_author.csv"
PREDICTIONS_PATH = "/content/drive/MyDrive/LogLLM/BGL/test_preds_original.csv"

# Load
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH)

print("Loading predictions...")
predictions = pd.read_csv(PREDICTIONS_PATH)

print("\nEmbeddings shape:", embeddings.shape)
print("Metadata shape:", metadata.shape)
print("Predictions shape:", predictions.shape)

# Check Alignment
assert len(embeddings) == len(metadata), \
    "Number of embeddings does not match metadata rows."

print("\nEmbedding and metadata lengths match.")

print("\nItem label distribution:")
print(metadata["item_label"].value_counts())

print("\nWindow label distribution:")
print(metadata["window_label"].value_counts())

print("\nPredicted window label distribution:")
print(predictions["Predicted_Label"].value_counts())

# ============================================================
# Map window-level predictions to individual logs
# ============================================================

# Each metadata row corresponds to one individual log.
# window_id identifies which original window that log belongs to.

prediction_map = predictions["Predicted_Label"].to_numpy()

assert metadata["window_id"].max() < len(prediction_map), \
    "Some metadata window IDs do not exist in predictions."

metadata["predicted_window_label"] = metadata["window_id"].map(
    lambda x: prediction_map[int(x)]
)


# Optional: PCA before UMAP
# Reducing 768 dimensions to 50 before UMAP makes UMAP faster and removes some noise.

from sklearn.decomposition import PCA

print("\nRunning PCA...")

pca = PCA(n_components=50, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

print("Variance explained by 50 PCA components:", pca.explained_variance_ratio_.sum())

# UMAP
print("\nRunning UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
)

embedding_2d = reducer.fit_transform(embeddings_pca)

print("UMAP complete.")

# Save UMAP Coordinates
umap_results = metadata.copy()

umap_results["UMAP_1"] = embedding_2d[:, 0]
umap_results["UMAP_2"] = embedding_2d[:, 1]

umap_results.to_csv(UMAP_OUTPUT, index=False)

print("\nSaved UMAP coordinates to:")
print(UMAP_OUTPUT)


# Plot 1: True individual-log labels
plt.figure(figsize=(10, 8))

normal = metadata["item_label"].to_numpy() == 0
anomaly = metadata["item_label"].to_numpy() == 1

plt.scatter(
    embedding_2d[normal, 0],
    embedding_2d[normal, 1],
    s=4,
    alpha=0.3,
    label="Normal"
)

plt.scatter(
    embedding_2d[anomaly, 0],
    embedding_2d[anomaly, 1],
    s=12,
    alpha=0.8,
    label="Anomalous"
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of BERT Encoder Embeddings — BGL")
plt.legend()
plt.tight_layout()

plt.show()


# Plot 2: LogLLM window-level predictions
predicted = metadata["predicted_window_label"].to_numpy()

plt.figure(figsize=(10, 8))

pred_normal = predicted == 0
pred_anomaly = predicted == 1

plt.scatter(
    embedding_2d[pred_normal, 0],
    embedding_2d[pred_normal, 1],
    s=4,
    alpha=0.3,
    label="Predicted Normal Window"
)

plt.scatter(
    embedding_2d[pred_anomaly, 0],
    embedding_2d[pred_anomaly, 1],
    s=12,
    alpha=0.8,
    label="Predicted Anomalous Window"
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of BERT Encoder Embeddings — LogLLM Predictions")
plt.legend()
plt.tight_layout()

plt.show()