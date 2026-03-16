import argparse
import pandas as pd
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# Config
# -------------------------------

MODEL_NAME = "google/flan-t5-base"
BATCH_SIZE = 32

# -------------------------------
# Load model
# -------------------------------

print("Loading FLAN-T5 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Using device:", device)


# -------------------------------
# Prompt builder
# -------------------------------

def build_prompt(log_text):

    return f"""
Rewrite this system log message.

Requirements:
- Keep the same meaning
- Change wording and structure
- Keep it concise like a real system log

Log message:
{log_text}

Rewritten log:
"""


# -------------------------------
# Batched paraphrasing
# -------------------------------

def batch_paraphrase(log_list):

    prompts = [build_prompt(log) for log in log_list]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        temperature=1.1,
        top_p=0.9,
        top_k=50
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return [x.strip() for x in decoded]


# -------------------------------
# Main
# -------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    print("Loading dataset:", args.input)

    df = pd.read_csv(args.input)

    print("Dataset size:", len(df))

    # ------------------------------------------------
    # Collect all logs that need rewriting
    # ------------------------------------------------

    logs_to_rewrite = []
    mapping = []   # (row_index, log_position)

    for idx, row in df.iterrows():

        logs = row["Content"].split(" ;-; ")
        labels = ast.literal_eval(row["item_Label"])

        for pos, (log, label) in enumerate(zip(logs, labels)):
            if label == 1:
                logs_to_rewrite.append(log)
                mapping.append((idx, pos))

    print("Logs to rewrite:", len(logs_to_rewrite))

    # ------------------------------------------------
    # Generate paraphrases in batches
    # ------------------------------------------------

    rewritten_logs = []

    for i in tqdm(range(0, len(logs_to_rewrite), BATCH_SIZE)):

        batch = logs_to_rewrite[i:i+BATCH_SIZE]

        outputs = batch_paraphrase(batch)

        rewritten_logs.extend(outputs)

    # ------------------------------------------------
    # Insert rewritten logs back into dataset
    # ------------------------------------------------

    new_contents = df["Content"].tolist()

    for (row_idx, log_pos), new_log in zip(mapping, rewritten_logs):

        logs = new_contents[row_idx].split(" ;-; ")
        logs[log_pos] = new_log
        new_contents[row_idx] = " ;-; ".join(logs)

    df["Para_Content"] = new_contents

    print("Saving dataset...")

    df.to_csv(args.output, index=False)

    print("Saved to:", args.output)


if __name__ == "__main__":
    main()