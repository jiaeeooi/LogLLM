import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, BitsAndBytesConfig,)
from peft import PeftModel

dataset_name = "BGL"
# Options:
#   "author"      - original author's trained BERT
#   "reproduced"  - reproduced baseline's trained BERT
#   "bert"        - BERT replacement
#   "sbert"       - SBERT replacement
#   "bge"         - BGE-M3 replacement
#   "qwen"        - Qwen replacement
encoder_name = "author"
# Options:
#   "encoder"     - embeddings before the projector
#   "projector"   - embeddings after the projector
embedding = "projector"

# -------------------------------------------

if embedding not in ["encoder", "projector"]:
    raise ValueError(f"Unknown embedding type: {embedding}")

DATA_DIR = Path(f"/content/drive/MyDrive/LogLLM/{dataset_name}")
ROOT_DIR = Path(__file__).parent
TRAIN_PATH = DATA_DIR / "train_probe.csv"
TEST_PATH = DATA_DIR / "test_probe.csv"

RESULTS_DIR = Path(f"/content/drive/MyDrive/LogLLM/{dataset_name}/lr_probe")
OUTPUT_DIR = RESULTS_DIR / encoder_name / embedding  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_CONFIG = {
    "author": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
        "output_name": "author",
    },

    "reproduced": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
        "output_name": "reproduced",
    },

    "bert": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
        "output_name": "bert",
    },

    "sbert": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "pooling": "sbert_mean",
        "output_name": "sbert",
    },

    "bge": {
        "model_name": "BAAI/bge-m3",
        "pooling": "bge_cls",
        "output_name": "bge",
    },

    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pooling": "qwen_last",
        "output_name": "qwen",
    },
}

config = ENCODER_CONFIG[encoder_name]

model_name = config["model_name"]
pooling = config["pooling"]
output_name = config["output_name"]

if encoder_name == "author":
    ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
elif encoder_name == "reproduced":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}"
elif encoder_name == "bert":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_bert"
elif encoder_name == "sbert":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_mpnet"
elif encoder_name == "bge":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_bgem3"
elif encoder_name == "qwen":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_qwen"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def load_encoder():
    if encoder_name in ["author", "reproduced"]:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        bert = BertModel.from_pretrained("bert-base-uncased", quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="cuda:0" if torch.cuda.is_available() else None)

        bert_ft_path = os.path.join(ft_path, "Bert_ft")

        if not os.path.exists(bert_ft_path):
            raise FileNotFoundError(f"BERT LoRA checkpoint not found:\n{bert_ft_path}")

        bert = PeftModel.from_pretrained(
            bert,
            bert_ft_path,
            is_trainable=False,
            torch_dtype=torch.float16
        )

        bert.eval()
        return tokenizer, bert

    elif encoder_name == "bert":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        bert = BertModel.from_pretrained("bert-base-uncased", quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="cuda:0" if torch.cuda.is_available() else None)
        
        bert.eval()
        return tokenizer, bert

    elif encoder_name == "sbert":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="cuda:0" if torch.cuda.is_available() else None,
        )

        model.eval()
        return tokenizer, model

    elif encoder_name == "bge":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        model.eval()
        return tokenizer, model

    elif encoder_name == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        model.eval()
        return tokenizer, model

def load_projector():
    projector_path = os.path.join(ft_path, "projector.pt")

    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Projector checkpoint not found:\n{projector_path}")

    # Load the trained projector weights
    projector_state = torch.load(projector_path, map_location="cpu", weights_only=True)

    # Infer dimensions from the trained checkpoint
    encoder_dim = encoder.config.hidden_size
    projector_dim = projector_state["weight"].shape[0]

    if projector_state["weight"].shape[1] != encoder_dim:
        raise ValueError(f"Projector input dimension ({projector_state['weight'].shape[1]}) doesn't match encoder dimension ({encoder_dim}).")

    projector = torch.nn.Linear(encoder_dim, projector_dim)

    projector.load_state_dict(projector_state)
    projector = projector.to(device)
    projector.eval()

    print(f"Projector checkpoint: {projector_path}")
    print(f"Projector dimensions: {encoder_dim} -> {projector_dim}")

    return projector


def extract_embeddings(logs, batch_size=32):
    all_embeddings = []

    encoder.eval()

    for start_idx in tqdm(range(0, len(logs), batch_size),desc=f"Extracting {encoder_name} embeddings"):
        batch_logs = logs[start_idx:start_idx + batch_size]
        inputs = tokenizer(batch_logs, return_tensors="pt", max_length=100, padding=True, truncation=True) # tokenizer
        inputs = {key: value.to(device) for key, value in inputs.items()} # Move to gpu
        
        # Forward Pass
        with torch.no_grad():
            encoder_outputs = encoder(**inputs)
            if pooling == "bert_cls":
                embeddings = encoder_outputs.pooler_output
            elif pooling == "sbert_mean":
                last_hidden = encoder_outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (
                    (last_hidden * attention_mask).sum(dim=1)
                    / attention_mask.sum(dim=1).clamp(min=1)
                )
            elif pooling == "bge_cls":
                embeddings = encoder_outputs.last_hidden_state[:, 0]
            elif pooling == "qwen_last":
                last_hidden = encoder_outputs.last_hidden_state
                sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
                embeddings = last_hidden[
                    torch.arange(last_hidden.size(0), device=last_hidden.device), 
                    sequence_lengths
                ]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            embeddings = embeddings.float().cpu().numpy()

        all_embeddings.append(embeddings)

    # Combine all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings

# Load and validate probe datasets
def load_probe_data(path, split_name):
    if not path.exists():
        raise FileNotFoundError(f"{split_name} probe dataset not found:\n{path}")

    df = pd.read_csv(path)
    required_columns = {"content", "item_label"}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"{split_name} dataset is missing columns: {missing_columns}")
    if df.empty:
        raise ValueError(f"{split_name} probe dataset is empty.")
    if df["content"].isna().any():
        raise ValueError(f"{split_name} dataset contains missing content.")
    if df["item_label"].isna().any():
        raise ValueError(f"{split_name} dataset contains missing labels.")

    logs = df["content"].astype(str).tolist()

    numeric_labels = pd.to_numeric(df["item_label"], errors="raise")

    if not np.all(numeric_labels.to_numpy() % 1 == 0):
        raise ValueError(f"{split_name} labels must be integers.")

    labels = numeric_labels.to_numpy(dtype=np.int64)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(
            f"{split_name} contains labels outside {{0, 1}}: "
            f"{np.unique(labels)}"
        )

    print(f"\n{split_name} dataset")
    print("-" * 50)
    print(f"Number of logs: {len(logs)}")
    print(
        "Label distribution:",
        dict(
            zip(
                *np.unique(labels, return_counts=True)
            )
        ),
    )

    return logs, labels

# Save embeddings and labels together
def save_embeddings(split_name, embeddings, labels, logs):
    if len(embeddings) != len(labels):
        raise ValueError(
            f"{split_name}: embedding count "
            f"({len(embeddings)}) does not match label count "
            f"({len(labels)})."
        )
    if len(embeddings) != len(logs):
        raise ValueError(
            f"{split_name}: embedding count "
            f"({len(embeddings)}) does not match log count "
            f"({len(logs)})."
        )

    output_path = (OUTPUT_DIR / f"{split_name}_embeddings.npz")
    np.savez_compressed(output_path, embeddings=embeddings, labels=labels, content=np.asarray(logs, dtype=str),)

    print(f"\nSaved {split_name} embeddings:")
    print(f"  Path: {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Labels: {labels.shape}")

    return output_path

def main():
    print("=" * 70)
    print("LOGLLM LOGISTIC REGRESSION PROBE - EMBEDDING EXTRACTION")
    print("=" * 70)
    print(f"Dataset       : {dataset_name}")
    print(f"Encoder       : {encoder_name}")
    print(f"Model         : {model_name}")
    print(f"Pooling       : {pooling}")
    print(f"Checkpoint    : {ft_path}")
    print(f"Train data    : {TRAIN_PATH}")
    print(f"Test data     : {TEST_PATH}")
    print(f"Output dir    : {OUTPUT_DIR}")
    print("=" * 70)

    # Load both splits first
    train_logs, train_labels = load_probe_data(TRAIN_PATH, "Train")

    test_logs, test_labels = load_probe_data(TEST_PATH, "Test",)

    # Check for accidental template overlap
    overlap = set(train_logs).intersection(test_logs)

    if overlap:
        raise ValueError(f"Found {len(overlap)} log templates in both probe train and test datasets. Resolve before extracting.")

    print("\nTrain/test template overlap check passed.")

    # Load the selected encoder only once
    global tokenizer, encoder
    tokenizer, encoder = load_encoder()

    print("\nEncoder loaded successfully.")

    # Extract train embeddings
    print("\n" + "=" * 70)
    print("EXTRACTING TRAIN EMBEDDINGS")
    print("=" * 70)

    train_embeddings = extract_embeddings(train_logs, batch_size=32)
    train_output_path = save_embeddings("train", train_embeddings, train_labels, train_logs)

    # Extract test embeddings using the same encoder
    print("\n" + "=" * 70)
    print("EXTRACTING TEST EMBEDDINGS")
    print("=" * 70)

    test_embeddings = extract_embeddings(test_logs, batch_size=32)
    test_output_path = save_embeddings("test", test_embeddings, test_labels, test_logs)

    # Final consistency checks
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError("Train and test embedding dimensions do not match.")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Encoder: {encoder_name}")
    print(f"Pooling: {pooling}")
    print(f"Train embedding shape: {train_embeddings.shape}")
    print(f"Test embedding shape:  {test_embeddings.shape}")
    print("\nFiles saved:")
    print(f"  {train_output_path}")
    print(f"  {test_output_path}")
    print("=" * 70)

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
