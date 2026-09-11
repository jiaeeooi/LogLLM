import os 
import re 
import ast 
from pathlib import Path

import numpy as np 
import pandas as pd 
import torch 
from tqdm import tqdm 
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig
from peft import PeftModel

dataset_name = "BGL" 
data_path = "/content/drive/MyDrive/LogLLM/BGL/bgl_test.csv"
Bert_path = "bert-base-uncased"
ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_2"
configuration = "orginal"
pooling = "cls" 

max_content_len = 100 
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir = "/content/drive/MyDrive/LogLLM/results/" f"embeddings_{dataset_name}"
os.makedirs(output_dir, exist_ok=True) 
embedding_path = os.path.join( output_dir, f"embeddings_{configuration}.npy" ) 
metadata_path = os.path.join( output_dir, f"embedding_metadata_{configuration}.csv" )

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

def load_bert():
    tokenizer = BertTokenizerFast.from_pretrained(Bert_path, do_lower_case=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load the model into memory using 4-bit precision
        bnb_4bit_use_double_quant=False,  # use double quantition
        bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
        bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
    )

    bert = BertModel.from_pretrained(Bert_path, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map=device)
    if ft_path is not None:
        Bert_ft_path = os.path.join(ft_path, 'Bert_ft')
        bert = PeftModel.from_pretrained(
            bert,
            Bert_ft_path,
            is_trainable=False,
            torch_dtype=torch.float16
        )

    bert.eval()
    return tokenizer, bert

def load_logs(): 
    df = pd.read_csv(data_path) 
    all_logs = [] 
    metadata = [] 
    for window_id, row in df.iterrows(): 
        # Content contains 100 logs separated by ' ;-; ' 
        logs = str(row["Content"]).split(" ;-; ") 
        # item_Label contains the label of every individual log 
        item_labels_raw = row["item_Label"] 
        try: 
            item_labels = ast.literal_eval(item_labels_raw) 
          
        except (ValueError, SyntaxError):
            # Handle cases where pandas already read it as a list 
            item_labels = item_labels_raw 
        
        if len(logs) != len(item_labels): 
            print( 
                f"WARNING: window {window_id} has " 
                f"{len(logs)} logs but " 
                f"{len(item_labels)} item labels." 
            ) 
        
        for log_id, log in enumerate(logs): 
            # Apply exactly the same preprocessing 
            log = replace_patterns(log) 
            all_logs.append(log) 
            metadata.append({ 
                "window_id": window_id, 
                "log_id": log_id, 
                "item_label": int(item_labels[log_id]), 
                "window_label": int(row["Label"]), 
                "content": log 
            }) 
                
    metadata_df = pd.DataFrame(metadata) 
    print(f"\nNumber of windows: {len(df)}") 
    print(f"Number of individual logs: {len(all_logs)}") 
    return all_logs, metadata_df

def extract_embeddings(tokenizer, bert, logs): 
    all_embeddings = [] 
    print("\nExtracting embeddings...") 
    for start in tqdm(range(0, len(logs), batch_size)): 
        batch_logs = logs[start:start + batch_size] 
        inputs = tokenizer(batch_logs, return_tensors="pt", max_length=max_content_len, padding=True, truncation=True) 
        inputs = {key: value.to(device) for key, value in inputs.items()} 
        with torch.no_grad(): 
            bert_outputs = bert(**inputs) 
            if pooling == "cls": 
                embeddings = bert_outputs.pooler_output 
            elif pooling == "mean": 
                hidden = bert_outputs.last_hidden_state 
                attention_mask = ( inputs["attention_mask"] .unsqueeze(-1) .expand(hidden.size()) .float() ) 
                sum_embeddings = ( hidden * attention_mask ).sum(dim=1) 
                sum_mask = attention_mask.sum(dim=1) 
                embeddings = ( sum_embeddings / sum_mask.clamp(min=1e-9) ) 
            else:
                raise ValueError( f"Unknown pooling method: {pooling}. " f"Use 'cls' or 'mean'." ) 
            embeddings = embeddings.float().cpu().numpy() 
            all_embeddings.append(embeddings) 
    embeddings = np.concatenate( all_embeddings, axis=0 ) 
    return embeddings






