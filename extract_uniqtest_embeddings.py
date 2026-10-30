import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, BitsAndBytesConfig,)
from peft import PeftModel
import gc

dataset_name = "BGL"       # "BGL" or "Liberty"
encoder_name = "author"    # author, reproduced, bert, sbert, bge, qwen
embedding = "encoder"      # "encoder" or "projector"

# -------------------------------------------

if embedding not in ["encoder", "projector"]:
    raise ValueError(f"Unknown embedding type: {embedding}")

DATA_DIR = Path(f"/content/drive/MyDrive/LogLLM/{dataset_name}")
ROOT_DIR = Path(__file__).parent

TEST_PATH = DATA_DIR / "test_deduplicated.csv"

RESULTS_DIR = Path(f"/content/drive/MyDrive/LogLLM/{dataset_name}/representation_analysis")
OUTPUT_DIR = RESULTS_DIR / encoder_name / embedding
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_CONFIG = {
    "author": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
        "output_name": "author"
    },
    "reproduced": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
        "output_name": "reproduced"
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "pooling": "bert_cls",
        "output_name": "bert"
    },
    "sbert": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "pooling": "sbert_mean",
        "output_name": "sbert"
    },
    "bge": {
        "model_name": "BAAI/bge-m3",
        "pooling": "bge_cls",
        "output_name": "bge"
    },
    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pooling": "qwen_last",
        "output_name": "qwen"
    }
}

if encoder_name not in ENCODER_CONFIG:
    raise ValueError(f"Unknown encoder_name: {encoder_name}")

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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_encoder():
    config = ENCODER_CONFIG[encoder_name]
    model_name = config["model_name"]

    if encoder_name in ["author", "reproduced"]:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        encoder = BertModel.from_pretrained(model_name, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="cuda:0" if torch.cuda.is_available() else None)

        encoder = PeftModel.from_pretrained(
            encoder,
            os.path.join(ft_path, "Bert_ft"),
            is_trainable=False,
            torch_dtype=torch.float16
        )
        encoder.eval()

    elif encoder_name == "bert":
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        encoder = BertModel.from_pretrained(model_name, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="cuda:0" if torch.cuda.is_available() else None)
        encoder.eval()

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            trust_remote_code=True if encoder_name in ["bge", "qwen"] else False
        )
        encoder.eval()

    return tokenizer, encoder

def load_projector(encoder):
    projector_path = os.path.join(ft_path, "projector.pt")

    projector_state = torch.load(projector_path, map_location="cpu", weights_only=True)

    encoder_dim = encoder.config.hidden_size
    projector_dim = projector_state["weight"].shape[0]

    if projector_state["weight"].shape[1] != encoder_dim:
        raise ValueError(f"Projector input dimension {projector_state['weight'].shape[1]} does not match encoder dimension {encoder_dim}")

    projector = torch.nn.Linear(encoder_dim, projector_dim)

    projector.load_state_dict(projector_state)
    projector = projector.to(device)
    projector.eval()

    return projector

def extract_embeddings(logs, tokenizer, encoder, projector=None, batch_size=32):
    all_embeddings = []
    pooling = ENCODER_CONFIG[encoder_name]["pooling"]
    encoder.eval()

    for start_idx in tqdm(range(0, len(logs), batch_size), desc=f"Extracting {encoder_name} embeddings"):
        batch_logs = logs[start_idx:start_idx + batch_size]
        inputs = tokenizer(batch_logs, return_tensors="pt", max_length=100, padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

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

            if embedding == "projector":
                if projector is None:
                    raise ValueError("Projector has not been loaded.")
                embeddings = projector(embeddings.float())
                embeddings = embeddings.half()

            embeddings = embeddings.float().cpu().numpy()

        all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

def load_test_data(path):
    df = pd.read_csv(path)
    required_columns = ["content", "item_label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")
    if df.empty:
        raise ValueError("Test dataset is empty.")
    if df[required_columns].isnull().any().any():
        raise ValueError("Test dataset contains missing values.")

    logs = df["content"].astype(str).tolist()
    numeric_labels = pd.to_numeric(df["item_label"], errors="raise")

    if not np.all(numeric_labels.to_numpy() % 1 == 0):
        raise ValueError("Test labels must be integers.")

    labels = numeric_labels.to_numpy(dtype=np.int64)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")
    if len(logs) != len(labels):
        raise ValueError("Number of logs and labels do not match.")
    if df["content"].duplicated().any():
        raise ValueError("Test dataset contains duplicate log messages.")

    print(f"Number of unique test logs: {len(logs)}")
    print(f"Normal logs: {(labels == 0).sum()}")
    print(f"Anomalous logs: {(labels == 1).sum()}")

    return logs, labels

def save_embeddings(embeddings, labels, logs):
    output_path = OUTPUT_DIR / "test_embeddings.npz"
    np.savez_compressed(output_path, embeddings=embeddings, labels=labels, content=np.asarray(logs, dtype=str))
    print(f"Saved embeddings to: {output_path}")

def main():
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Encoder: {encoder_name}")
    print(f"Embedding: {embedding}")
    print("=" * 60)

    logs, labels = load_test_data(TEST_PATH)
    tokenizer, encoder = load_encoder()
    projector = None

    if embedding == "projector":
        projector = load_projector(encoder)

    # Extract embeddings
    embeddings = extract_embeddings(logs, tokenizer, encoder, projector)

    print(f"Embedding shape: {embeddings.shape}")

    if len(embeddings) != len(labels):
        raise ValueError("Number of embeddings does not match number of labels.")

    # Save embeddings
    save_embeddings(embeddings, labels, logs)

    print("Representation embedding extraction completed.")

    # Clean up
    del tokenizer, encoder, projector, embeddings
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()