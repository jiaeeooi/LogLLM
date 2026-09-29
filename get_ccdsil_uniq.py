import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Paths
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

# Original label distribution
print("\nOriginal item label distribution:")
print(metadata["item_label"].value_counts())

print("\nFinding unique templates...")

# Keep first occurrence of every unique processed log message
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

# Expected BGL numbers
assert len(metadata) == 371, \
    f"Expected 371 unique templates, found {len(metadata)}"

assert (metadata["item_label"] == 0).sum() == 334
assert (metadata["item_label"] == 1).sum() == 37

print("\nAll unique-template checks passed.")

# Separate normal and anomaly embeddings
labels = metadata["item_label"].to_numpy()

normal = labels == 0
anomaly = labels == 1

normal_embeddings = embeddings[normal]
anomaly_embeddings = embeddings[anomaly]

print("\nNormal templates:", normal_embeddings.shape[0])
print("Anomalous templates:", anomaly_embeddings.shape[0])

# Calculate centroids
print("\nCalculating class centroids...")

normal_centroid = normal_embeddings.mean(axis=0)
anomaly_centroid = anomaly_embeddings.mean(axis=0)


# Centroid cosine similarity / distance
centroid_cosine_similarity = cosine_similarity(
    normal_centroid.reshape(1, -1),
    anomaly_centroid.reshape(1, -1)
)[0, 0]

centroid_cosine_distance = 1 - centroid_cosine_similarity

print("\n" + "=" * 60)
print("CENTROID ANALYSIS")
print("=" * 60)

print(f"Normal centroid dimension:      {normal_centroid.shape[0]}")
print(f"Anomaly centroid dimension:     {anomaly_centroid.shape[0]}")

print(f"\nCentroid cosine similarity:     {centroid_cosine_similarity:.6f}")
print(f"Centroid cosine distance:       {centroid_cosine_distance:.6f}")

# Silhouette score
print("\nCalculating silhouette score...")

silhouette = silhouette_score(
    embeddings,
    labels,
    metric="cosine"
)

print("\n" + "=" * 60)
print("SILHOUETTE ANALYSIS")
print("=" * 60)

print(f"Silhouette score (cosine):      {silhouette:.6f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Unique templates:               {len(embeddings):,}")
print(f"Normal templates:               {normal.sum():,}")
print(f"Anomalous templates:            {anomaly.sum():,}")
print(f"Centroid cosine similarity:     {centroid_cosine_similarity:.6f}")
print(f"Centroid cosine distance:       {centroid_cosine_distance:.6f}")
print(f"Silhouette score (cosine):      {silhouette:.6f}")