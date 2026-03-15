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
        max_length=64,
        do_sample=True,
        top_k=120,
        top_p=0.95
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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