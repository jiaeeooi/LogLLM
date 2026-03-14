import pandas as pd
import ast
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


# ==============================
# Argument Parser
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input CSV dataset")
parser.add_argument("--output", type=str, required=True, help="Output CSV file")

args = parser.parse_args()


# ==============================
# Load Paraphrasing Model
# ==============================
model_name = "Vamsi/T5_Paraphrase_Paws"

print("Loading paraphrasing model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Using device:", device)


# ==============================
# Paraphrase Function
# ==============================
def paraphrase_log(log_text):

    prompt = "paraphrase: " + log_text + " </s>"

    encoding = tokenizer(
        prompt,
        padding="longest",
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        num_return_sequences=1,
        early_stopping=True
    )

    paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("ORIGINAL:", log_text)
    print("PARAPHRASE:", paraphrase)
    print()

    return paraphrase


# ==============================
# Paraphrase Window
# ==============================
def paraphrase_window(row):

    logs = row["Content"].split(" ;-; ")
    labels = ast.literal_eval(row["item_Label"])

    new_logs = []

    for log, label in zip(logs, labels):

        if label == 1:
            new_log = paraphrase_log(log)
        else:
            new_log = log

        new_logs.append(new_log)

    return " ;-; ".join(new_logs)


# ==============================
# Load Dataset
# ==============================
print("Loading dataset:", args.input)

df = pd.read_csv(args.input)

print("Dataset size:", len(df))


# ==============================
# Generate Paraphrases
# ==============================
print("Generating paraphrases...")

tqdm.pandas()

df["Para_Content"] = df.progress_apply(paraphrase_window, axis=1)


# ==============================
# Save Dataset
# ==============================
df.to_csv(args.output, index=False)

print("Saved paraphrased dataset to:", args.output)