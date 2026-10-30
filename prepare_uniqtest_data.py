import ast
import re
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

DATASETS = {
    "BGL": {
        "test": "/content/drive/MyDrive/LogLLM/BGL/bgl_test.csv",
        "output": "/content/drive/MyDrive/LogLLM/BGL/test_deduplicated.csv"
    },
    "Liberty": {
        "test": "/content/drive/MyDrive/LogLLM/Liberty/test.csv",
        "output": "/content/drive/MyDrive/LogLLM/Liberty/test_deduplicated.csv"
    }
}

DELIMITER = " ;-; "

patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',
]

combined_pattern = '|'.join(patterns)

def replace_patterns(text):
    text = re.sub(r'[\.]{3,}', '.. ', text)
    text = re.sub(combined_pattern, '<*>', text)
    return text

def prepare_test_set(csv_path, output_path, dataset_name):
    df = pd.read_csv(csv_path)

    required_columns = {"Content", "item_Label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{dataset_name} CSV is missing columns: {missing_columns}")

    logs = []

    for window_id, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
        window_logs = str(row["Content"]).split(DELIMITER)
        item_labels = ast.literal_eval(row["item_Label"])
        item_labels = [int(label) for label in item_labels]

        if len(window_logs) != len(item_labels):
            raise ValueError(
                f"{dataset_name}, window {window_id}: "
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

    print(f"\n{dataset_name} - Before deduplication:")
    print(f"  Original windows: {len(df):,}")
    print(f"  Individual logs:  {len(result):,}")

    # Check for conflicting labels after normalization
    label_counts = result.groupby("content")["item_label"].nunique()

    conflicting_templates = label_counts[label_counts > 1]

    if len(conflicting_templates) > 0:
        print(f"\nFound {len(conflicting_templates):,} messages with conflicting labels:")

        for template in conflicting_templates.index:
            subset = result[result["content"] == template]
            print(f"\nMessage: {template}")
            print(subset["item_label"].value_counts().sort_index())

        result = result[~result["content"].isin(conflicting_templates.index)].copy()
        print(f"\nExcluded {len(conflicting_templates):,} conflicting unique messages.")

    # Remove duplicate messages
    deduplicated = result.drop_duplicates(subset=["content"], keep="first").copy()
    deduplicated = deduplicated[["content", "item_label"]].reset_index(drop=True)
    assert deduplicated["content"].is_unique

    print(f"\n{dataset_name} - After deduplication:")
    print(f"  Unique messages: {len(deduplicated):,}")

    print("\nLabel distribution:")
    print(deduplicated["item_label"].value_counts().sort_index())

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    deduplicated.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return deduplicated

bgl_test = prepare_test_set(DATASETS["BGL"]["test"], DATASETS["BGL"]["output"], "BGL")
liberty_test = prepare_test_set(DATASETS["Liberty"]["test"], DATASETS["Liberty"]["output"], "Liberty")
