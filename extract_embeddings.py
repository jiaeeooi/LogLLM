import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, BitsAndBytesConfig
from peft import PeftModel

dataset_name = "BGL"
# Options:
#   "author"
#   "reproduced"
#   "sbert"
#   "bge"
#   "qwen"
encoder_name = "bge"

# -------------------------------------------

ROOT_DIR = Path(__file__).parent
# Three dataset paths
DATA_PATHS = {
    "BGL": "/content/drive/MyDrive/LogLLM/BGL/bgl_test.csv",
    "Thunderbird": "/content/drive/MyDrive/LogLLM/Thunderbird/test.csv",
}

# Output directory
RESULTS_DIR = Path("/content/drive/MyDrive/LogLLM/results")
OUTPUT_DIR = RESULTS_DIR / f"embeddings_{dataset_name}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check Dataset
if dataset_name not in DATA_PATHS:
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Available datasets: {list(DATA_PATHS.keys())}"
    )
data_path = DATA_PATHS[dataset_name]

# Check Encoder
VALID_ENCODERS = [
    "author",
    "reproduced",
    "sbert",
    "bge",
    "qwen",
]
if encoder_name not in VALID_ENCODERS:
    raise ValueError(
        f"Unknown encoder '{encoder_name}'. "
        f"Available encoders: {VALID_ENCODERS}"
    )

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

# ft_path
if encoder_name == "author":
    ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
elif encoder_name == "reproduced":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}"
elif encoder_name == "sbert":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_mpnet"
elif encoder_name == "bge":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_bgem3"
elif encoder_name == "qwen":
    ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_qwen"

# Output File Path
embedding_path = (OUTPUT_DIR / f"embeddings_{output_name}.npy")
metadata_path = (OUTPUT_DIR / f"embedding_metadata_{output_name}.csv")

# Print
print("=" * 70)
print("Embedding Extraction Configuration")
print("=" * 70)
print(f"Dataset       : {dataset_name}")
print(f"Encoder       : {encoder_name}")
print(f"Model         : {model_name}")
print(f"Pooling       : {pooling}")
print(f"Data path     : {data_path}")
print(f"Checkpoint    : {ft_path}")
print(f"Output dir    : {OUTPUT_DIR}")
print(f"Embedding     : {embedding_path}")
print(f"Metadata      : {metadata_path}")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(data_path)
print(f"Number of windows: {len(df)}")
print(f"Columns: {list(df.columns)}")

patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?', #  IP
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',   # Special MAC
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',   # MAC
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',  # File Path
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # word have number
]

combined_pattern = '|'.join(patterns)

def replace_patterns(text):
    text = re.sub(r'[\.]{3,}', '.. ', text)    # Replace multiple '.' with '.. '
    text = re.sub(combined_pattern, '<*>', text)
    return text


# ============================================================
# 11. LOAD ENCODER
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_encoder():

    # --------------------------------------------------------
    # Author / Reproduced BERT
    # --------------------------------------------------------
    if encoder_name in ["author", "reproduced"]:

        print("\nLoading BERT tokenizer...")

        tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

        print("Loading BERT model...")

        bert = BertModel.from_pretrained(
            "bert-base-uncased",
            quantization_config=bnb_config,
            device_map="cuda:0" if torch.cuda.is_available() else None,
        )

        # ----------------------------------------------------
        # Load LoRA weights from LogLLM checkpoint
        # ----------------------------------------------------
        bert_ft_path = os.path.join(
            ft_path,
            "Bert_ft"
        )

        print(f"\nLoading BERT LoRA checkpoint:")
        print(bert_ft_path)

        if not os.path.exists(bert_ft_path):
            raise FileNotFoundError(
                f"BERT LoRA checkpoint not found:\n{bert_ft_path}"
            )

        bert = PeftModel.from_pretrained(
            bert,
            bert_ft_path
        )

        bert.eval()

        return tokenizer, bert


    # --------------------------------------------------------
    # SBERT / MPNet
    # --------------------------------------------------------
    elif encoder_name == "sbert":

        print("\nLoading SBERT / MPNet tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        print("Loading SBERT / MPNet model...")

        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0" if torch.cuda.is_available() else None,
        )

        model.eval()

        return tokenizer, model


    # --------------------------------------------------------
    # BGE-M3
    # --------------------------------------------------------
    elif encoder_name == "bge":

        print("\nLoading BGE-M3 tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print("Loading BGE-M3 model...")

        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        model.eval()

        return tokenizer, model


    # --------------------------------------------------------
    # Qwen3-Embedding-0.6B
    # --------------------------------------------------------
    elif encoder_name == "qwen":

        print("\nLoading Qwen tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print("Loading Qwen model...")

        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        model.eval()

        return tokenizer, model


# ============================================================
# 12. LOAD MODEL
# ============================================================

tokenizer, encoder = load_encoder()

print("\nEncoder loaded successfully.")


# ============================================================
# 13. EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(logs, batch_size=32):

    all_embeddings = []
    metadata = []

    encoder.eval()

    for start_idx in tqdm(
        range(0, len(logs), batch_size),
        desc=f"Extracting {encoder_name} embeddings"
    ):

        batch_logs = logs[start_idx:start_idx + batch_size]

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------
        inputs = tokenizer(
            batch_logs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Move inputs to GPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------
        with torch.no_grad():

            encoder_outputs = encoder(**inputs)

            # =================================================
            # BERT CLS / pooler_output
            # =================================================
            if pooling == "bert_cls":

                embeddings = encoder_outputs.pooler_output


            # =================================================
            # SBERT / MPNet mean pooling
            # =================================================
            elif pooling == "sbert_mean":

                last_hidden = encoder_outputs.last_hidden_state

                attention_mask = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                )

                embeddings = (
                    (last_hidden * attention_mask).sum(dim=1)
                    /
                    attention_mask.sum(dim=1).clamp(min=1)
                )


            # =================================================
            # BGE-M3 CLS / first-token representation
            # =================================================
            elif pooling == "bge_cls":

                embeddings = (
                    encoder_outputs
                    .last_hidden_state[:, 0]
                )


            # =================================================
            # Qwen3 last valid token
            # =================================================
            elif pooling == "qwen_last":

                last_hidden = (
                    encoder_outputs.last_hidden_state
                )

                sequence_lengths = (
                    inputs["attention_mask"].sum(dim=1) - 1
                )

                embeddings = last_hidden[
                    torch.arange(
                        last_hidden.size(0),
                        device=last_hidden.device
                    ),
                    sequence_lengths
                ]


            else:
                raise ValueError(
                    f"Unknown pooling method: {pooling}"
                )

            # ------------------------------------------------
            # IMPORTANT:
            # No normalization before saving.
            # This keeps the encoder representation
            # consistent with the LogLLM encoder pipeline.
            # ------------------------------------------------
            embeddings = (
                embeddings
                .float()
                .cpu()
                .numpy()
            )

        all_embeddings.append(embeddings)


    # --------------------------------------------------------
    # Combine all batches
    # --------------------------------------------------------
    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    return all_embeddings


# ============================================================
# 14. PREPARE LOGS + METADATA
# ============================================================

print("\nPreparing logs...")

logs = []
metadata = []

for window_id, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processing windows"
):

    # --------------------------------------------------------
    # Each window contains multiple logs separated by:
    # " ;-; "
    # --------------------------------------------------------
    window_logs = str(row["Content"]).split(" ;-; ")

    item_labels = str(row["item_Label"]).split(" ;-; ")

    for log_id, log in enumerate(window_logs):

        log = replace_patterns(log)

        logs.append(log)

        # ----------------------------------------------------
        # Make sure label exists
        # ----------------------------------------------------
        if log_id < len(item_labels):
            item_label = int(item_labels[log_id])
        else:
            item_label = 0

        metadata.append({
            "window_id": window_id,
            "log_id": log_id,
            "item_label": item_label,
            "window_label": int(row["Label"]),
            "content": log
        })


print(f"\nTotal individual logs: {len(logs)}")


# ============================================================
# 15. EXTRACT EMBEDDINGS
# ============================================================

embeddings = extract_embeddings(
    logs,
    batch_size=32
)


# ============================================================
# 16. VERIFY ALIGNMENT
# ============================================================

print("\nChecking embedding / metadata alignment...")

assert len(embeddings) == len(metadata), (
    f"Mismatch: {len(embeddings)} embeddings vs "
    f"{len(metadata)} metadata rows"
)

print("Alignment check passed.")

print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# 17. SAVE EMBEDDINGS
# ============================================================

print("\nSaving embeddings...")

np.save(
    embedding_path,
    embeddings
)


# ============================================================
# 18. SAVE METADATA
# ============================================================

print("Saving metadata...")

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(f"Dataset          : {dataset_name}")
print(f"Encoder          : {encoder_name}")
print(f"Pooling          : {pooling}")
print(f"Number of logs   : {len(embeddings)}")
print(f"Embedding dim    : {embeddings.shape[1]}")

print("\nFiles saved:")

print(f"Embeddings:")
print(f"  {embedding_path}")

print(f"\nMetadata:")
print(f"  {metadata_path}")

print("=" * 70)