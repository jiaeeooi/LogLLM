# ==============================
# Imports
# ==============================
import pandas as pd
import ast
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
import random


# ==============================
# Arguments
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()


# ==============================
# Load Model
# ==============================
model_name = "Vamsi/T5_Paraphrase_Paws"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Using device:", device)

# ==============================
# Tame anomaly words 
# ==============================
def tame_anomaly_words(text):
    replacements = {
        r"\berror\b": ["issue", "problem"],
        r"\bfailure\b": ["condition", "event"],
        r"\bfailed\b": ["not completed", "unsuccessful"],
        r"\bfatal\b": ["critical", "serious"],
        r"\bcritical\b": ["notable", "significant"],
        r"\bpanic\b": ["alert", "warning"],
        r"\bfault\b": ["irregularity", "issue"],
        r"\bexception\b": ["event", "condition"],
        r"\binvalid\b": ["unexpected", "unusual"],
        r"\bdetected\b": ["observed", "identified", "found"],
        r"\binterrupt\b": ["signal", "event"],
        r"\bsevered\b": ["disconnected", "interrupted"],
        r"\bterminated\b": ["stopped", "ended"],
        r"\bmajor\b": ["notable", "significant"]
    }
    for pattern, options in replacements.items():
        text = re.sub(pattern, random.choice(options), text, flags=re.IGNORECASE)
    return text



# ==============================
# Load Dataset
# ==============================
print("Loading dataset...")

df = pd.read_csv(args.input)

print("Dataset size:", len(df))


# ==============================
# Step 1: Collect anomalous logs
# ==============================
print("Collecting anomalous logs...")

all_logs = []
log_positions = []   # (row_idx, log_idx)

for row_idx, row in tqdm(df.iterrows(), total=len(df)):

    logs = row["Content"].split(" ;-; ")
    labels = ast.literal_eval(row["item_Label"])

    for log_idx, (log, label) in enumerate(zip(logs, labels)):

        if label == 1:
            all_logs.append("paraphrase: " + log + " </s>")
            log_positions.append((row_idx, log_idx))

print("Total anomalous logs:", len(all_logs))


# ==============================
# Step 2: Batch paraphrasing
# ==============================
batch_size = 32
paraphrased_logs = []

print("Paraphrasing logs in batches...")

for i in tqdm(range(0, len(all_logs), batch_size)):

    batch = all_logs[i:i+batch_size]

    encoding = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    outputs = model.generate(
        **encoding,
        max_length=128,
        do_sample=True,
        temperature=1.2,
        top_k=50,
        top_p=0.85,
        repetition_penalty=1.2
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded = [tame_anomaly_words(text) for text in decoded]
    paraphrased_logs.extend(decoded)


# ==============================
# Step 3: Rebuild windows
# ==============================
print("Rebuilding windows...")

para_content = df["Content"].tolist()

for (row_idx, log_idx), new_log in zip(log_positions, paraphrased_logs):

    logs = para_content[row_idx].split(" ;-; ")
    logs[log_idx] = new_log
    para_content[row_idx] = " ;-; ".join(logs)

df["Para_Content"] = para_content


# ==============================
# Save
# ==============================
df.to_csv(args.output, index=False)
print("Saved to:", args.output)