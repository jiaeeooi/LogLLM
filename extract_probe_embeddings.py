import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, BitsAndBytesConfig
from peft import PeftModel


dataset_name = "BGL" # BGL or Liberty
encoder_name = "bert" # "author", "reproduced", "bert", "sbert", "bge", or "qwen"
DATA_DIR = Path("/content/drive/MyDrive/LogLLM/BGL")

TRAIN_PATH = DATA_DIR / "train_probe.csv"
TEST_PATH = DATA_DIR / "test_probe.csv"

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/LogLLM/results/lr_probe"
) / encoder_name

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 100
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_4bit = device.type == "cuda"

print(f"Dataset: {dataset_name}")
print(f"Encoder: {encoder_name}")
print(f"Device: {device}")
print(f"4-bit quantization: {use_4bit}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# 2. Encoder configurations
# ============================================================

ENCODER_CONFIG = {
    "bert": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
    },
    "sbert": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "pooling": "sbert_mean",
    },
    "bge": {
        "model_name": "BAAI/bge-m3",
        "pooling": "bge_cls",
    },
    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pooling": "qwen_last",
    },
}

if encoder_name not in ENCODER_CONFIG:
    raise ValueError(
        f"Unknown encoder: {encoder_name}. "
        f"Choose from {list(ENCODER_CONFIG.keys())}"
    )

config = ENCODER_CONFIG[encoder_name]
model_name = config["model_name"]
pooling = config["pooling"]

# ============================================================
# 3. Load encoder
# ============================================================

def load_encoder():
    print(f"\nLoading model: {model_name}")

    load_kwargs = {}

    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "cuda:0"

    if encoder_name == "bert":
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertModel.from_pretrained(
            model_name, **load_kwargs
        )

    elif encoder_name == "sbert":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, **load_kwargs
        )

    elif encoder_name == "bge":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            **load_kwargs,
        )

    elif encoder_name == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            **load_kwargs,
        )

    if not use_4bit:
        model = model.to(device)

    model.eval()

    print("Model loaded successfully.")
    return tokenizer, model


# ============================================================
# 4. Extract embeddings
# ============================================================

def extract_embeddings(contents, tokenizer, model):
    all_embeddings = []

    for start in tqdm(
        range(0, len(contents), BATCH_SIZE),
        desc=f"Extracting {encoder_name} embeddings",
    ):
        batch_contents = contents[start : start + BATCH_SIZE]

        inputs = tokenizer(
            batch_contents,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            if pooling == "bert_cls":
                # Same pooling as the original extraction script.
                embeddings = outputs.pooler_output

            elif pooling == "sbert_mean":
                # Attention-mask-weighted mean pooling.
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)

                embeddings = (
                    (hidden * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp(min=1)
                )

            elif pooling == "bge_cls":
                embeddings = outputs.last_hidden_state[:, 0, :]

            elif pooling == "qwen_last":
                # Last non-padding token, following the original script.
                hidden = outputs.last_hidden_state
                sequence_lengths = (
                    inputs["attention_mask"].sum(dim=1) - 1
                )

                embeddings = hidden[
                    torch.arange(
                        hidden.size(0), device=hidden.device
                    ),
                    sequence_lengths,
                ]

            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            all_embeddings.append(
                embeddings.float().cpu().numpy()
            )

    if not all_embeddings:
        raise ValueError("The dataset contains no logs to embed.")

    return np.concatenate(all_embeddings, axis=0)


# ============================================================
# 5. Load and validate probe data
# ============================================================

def load_probe_csv(path, split_name):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"content", "item_label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(f"{split_name} probe dataset is empty.")

    if df["content"].isna().any():
        raise ValueError(
            f"{split_name} contains missing log content."
        )

    if df["item_label"].isna().any():
        raise ValueError(
            f"{split_name} contains missing labels."
        )

    contents = df["content"].astype(str).tolist()

    # Reject non-integer labels rather than silently coercing them.
    numeric_labels = pd.to_numeric(
        df["item_label"], errors="raise"
    )

    if not np.all(numeric_labels.to_numpy() % 1 == 0):
        raise ValueError(
            f"{split_name} contains non-integer labels."
        )

    labels = numeric_labels.to_numpy(dtype=np.int64)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(
            f"{split_name} has labels outside {{0, 1}}: "
            f"{np.unique(labels)}"
        )

    print(f"\n{split_name} dataset:")
    print(f"  Number of logs: {len(contents)}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return contents, labels


# ============================================================
# 6. Main: process train and test in the same run
# ============================================================

def main():
    train_contents, train_labels = load_probe_csv(
        TRAIN_PATH, "Train"
    )
    test_contents, test_labels = load_probe_csv(
        TEST_PATH, "Test"
    )

    # The probe dataset should already be deduplicated and cleaned.
    # Check for accidental train-test overlap without changing content.
    overlap = set(train_contents).intersection(test_contents)

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} log templates in both train and "
            "test probe datasets. Check the probe dataset preparation "
            "before extracting embeddings."
        )

    tokenizer, model = load_encoder()

    # Extract train embeddings.
    train_embeddings = extract_embeddings(
        train_contents, tokenizer, model
    )

    if len(train_embeddings) != len(train_labels):
        raise RuntimeError(
            "Train embedding count does not match train label count."
        )

    train_output = OUTPUT_DIR / "train_embeddings.npz"

    np.savez_compressed(
        train_output,
        embeddings=train_embeddings,
        labels=train_labels,
        content=np.asarray(train_contents, dtype=str),
    )

    print(f"\nSaved train embeddings: {train_output}")
    print(f"  Embedding shape: {train_embeddings.shape}")

    # Extract test embeddings using the same loaded encoder.
    test_embeddings = extract_embeddings(
        test_contents, tokenizer, model
    )

    if len(test_embeddings) != len(test_labels):
        raise RuntimeError(
            "Test embedding count does not match test label count."
        )

    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise RuntimeError(
            "Train and test embedding dimensions do not match."
        )

    test_output = OUTPUT_DIR / "test_embeddings.npz"

    np.savez_compressed(
        test_output,
        embeddings=test_embeddings,
        labels=test_labels,
        content=np.asarray(test_contents, dtype=str),
    )

    print(f"\nSaved test embeddings: {test_output}")
    print(f"  Embedding shape: {test_embeddings.shape}")

    print("\nEmbedding extraction completed successfully.")
    print(f"Train output: {train_output}")
    print(f"Test output:  {test_output}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
