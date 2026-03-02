import pandas as pd
import ast

def paraphrase_log(log_text):
    # TODO: replace this with your LLM / paraphraser
    return your_paraphraser(log_text)

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


df["Para_Content"] = df.apply(paraphrase_window, axis=1)

df.to_csv("bgl_with_paraphrase.csv", index=False)