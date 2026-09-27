import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

PLOT_OUTPUT = "/content/drive/MyDrive/LogLLM/results/embeddings_BGL/umap_uniq_author.png"
EMBEDDINGS_PATH = "/content/drive/MyDrive/LogLLM/results/embeddings_BGL/embeddings_author.npy"
METADATA_PATH = "/content/drive/MyDrive/LogLLM/results/embeddings_BGL/embedding_metadata_author.csv"

# Load
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH)

print("\nEmbeddings shape:", embeddings.shape)
print("Metadata shape:", metadata.shape)


# Check alignment
assert len(embeddings) == len(metadata), \
    "Number of embeddings does not match metadata rows."

print("\nEmbedding and metadata lengths match.")

# Check original label distributionn
print("\nOriginal item label distribution:")
print(metadata["item_label"].value_counts())


# Find unique templates
print("\nFinding unique templates...")

# Keep the first occurrence of every unique processed log message
unique_mask = ~metadata["content"].duplicated(keep="first")

unique_idx = np.where(unique_mask)[0]

print(f"Total observations:       {len(metadata):,}")
print(f"Unique templates:         {len(unique_idx):,}")

# Select corresponding embeddings and metadata
embeddings = embeddings[unique_idx]
metadata = metadata.iloc[unique_idx].reset_index(drop=True)

print("\nUnique-template embeddings shape:", embeddings.shape)
print("Unique-template metadata shape:", metadata.shape)

# Verify template labels
print("\nUnique-template label distribution:")
print(metadata["item_label"].value_counts())

# Every template should have exactly one label
label_counts = (
    metadata.groupby("content")["item_label"]
    .nunique()
)

assert (label_counts == 1).all(), \
    "At least one template has multiple labels."

# Expected BGL numbers from your previous analysis
assert len(metadata) == 371, \
    f"Expected 371 unique templates, found {len(metadata)}"

assert (metadata["item_label"] == 0).sum() == 334
assert (metadata["item_label"] == 1).sum() == 37

print("\nAll unique-template checks passed.")

# UMAP
print("\nRunning UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42
)

embedding_2d = reducer.fit_transform(embeddings)

print("UMAP complete.")

# Plot
plt.figure(figsize=(10, 8))

normal = metadata["item_label"].to_numpy() == 0
anomaly = metadata["item_label"].to_numpy() == 1

plt.scatter(
    embedding_2d[normal, 0],
    embedding_2d[normal, 1],
    s=25,
    alpha=0.7,
    label="Normal"
)

plt.scatter(
    embedding_2d[anomaly, 0],
    embedding_2d[anomaly, 1],
    s=40,
    alpha=0.9,
    label="Anomalous"
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of Unique Log Templates — BGL")
plt.legend()
plt.tight_layout()

# Save
plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches="tight")

print("\nSaved UMAP plot to:")
print(PLOT_OUTPUT)

plt.close()