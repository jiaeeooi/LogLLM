import ast
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# File paths
#TRAIN_PATH = "/content/drive/MyDrive/LogLLM/BGL/bgl_train.csv"
#TEST_PATH = "/content/drive/MyDrive/LogLLM/BGL/bgl_test.csv"
TRAIN_PATH = "/content/drive/MyDrive/LogLLM/Liberty/train.csv"
TEST_PATH = "/content/drive/MyDrive/LogLLM/Liberty/test.csv"

#OUTPUT_TRAIN_PATH = "/content/drive/MyDrive/LogLLM/BGL/train_probe.csv"
#OUTPUT_TEST_PATH = "/content/drive/MyDrive/LogLLM/BGL/test_probe.csv"
OUTPUT_TRAIN_PATH = "/content/drive/MyDrive/LogLLM/Liberty/train_probe.csv"
OUTPUT_TEST_PATH = "/content/drive/MyDrive/LogLLM/Liberty/test_probe.csv"

DELIMITER = " ;-; "

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

def extract_individual_logs(csv_path, split_name):
    df = pd.read_csv(csv_path)

    required_columns = {"Content", "item_Label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"{split_name} CSV is missing columns: {missing_columns}")

    logs = []

    for window_id, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} windows"):
        window_logs = str(row["Content"]).split(DELIMITER)

        item_labels = ast.literal_eval(row["item_Label"])
        item_labels = [int(label) for label in item_labels]

        if len(window_logs) != len(item_labels):
            raise ValueError(
                f"{split_name}, window {window_id}: "
                f"{len(window_logs)} logs but "
                f"{len(item_labels)} item labels. "
                "Check the delimiter and item_Label format."
            )

        for log, item_label in zip(window_logs, item_labels):
            logs.append({
                "content": replace_patterns(log),
                "item_label": item_label
            })

    result = pd.DataFrame(logs, columns=["content", "item_label"])

    print(f"\n{split_name}:")
    print(f"  Original windows: {len(df):,}")
    print(f"  Individual logs:  {len(result):,}")

    return result

# Load and preprocess train/test
train_logs = extract_individual_logs(TRAIN_PATH, "TRAIN")
test_logs = extract_individual_logs(TEST_PATH, "TEST")

all_logs = pd.concat([train_logs.assign(split="train"),test_logs.assign(split="test")], ignore_index=True)

label_counts = all_logs.groupby("content")["item_label"].nunique()
conflicting_templates = set(label_counts[label_counts > 1].index)

print(f"Conflicting templates: {len(conflicting_templates)}")

for template in sorted(conflicting_templates):
    subset = all_logs[all_logs["content"] == template]
    print(f"\nTemplate: {template}")
    print(subset["item_label"].value_counts().sort_index())

train_before = len(train_logs)
test_before = len(test_logs)

train_logs = train_logs[~train_logs["content"].isin(conflicting_templates)].copy()
test_logs = test_logs[~test_logs["content"].isin(conflicting_templates)].copy()

print("\nAfter excluding conflicting templates:")
print(f"Train: {train_before:,} -> {len(train_logs):,}")
print(f"Test:  {test_before:,} -> {len(test_logs):,}")

train_probe = train_logs.drop_duplicates(subset=["content"], keep="first").copy()
test_probe = test_logs.drop_duplicates(subset=["content"],keep="first").copy()

train_templates = set(train_probe["content"])

overlap_mask = test_probe["content"].isin(train_templates)
overlap_count = int(overlap_mask.sum())

test_probe = test_probe.loc[~overlap_mask].copy()

print(f"\nTest templates removed due to train overlap: {overlap_count:,}")

train_probe = train_probe[["content", "item_label"]].reset_index(drop=True)
test_probe = test_probe[["content", "item_label"]].reset_index(drop=True)

assert train_probe["content"].is_unique
assert test_probe["content"].is_unique
assert set(train_probe["content"]).isdisjoint(set(test_probe["content"]))

print("\nFinal train label distribution:")
print(train_probe["item_label"].value_counts().sort_index())

print("\nFinal test label distribution:")
print(test_probe["item_label"].value_counts().sort_index())

train_probe.to_csv(OUTPUT_TRAIN_PATH, index=False)
test_probe.to_csv(OUTPUT_TEST_PATH, index=False)

print(f"\nSaved: {OUTPUT_TRAIN_PATH}")
print(f"Saved: {OUTPUT_TEST_PATH}")