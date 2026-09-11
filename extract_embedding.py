import os 
import re 
import ast 
from pathlib import Path

import numpy as np 
import pandas as pd 
import torch 
from tqdm import tqdm 
from transformers import BertTokenizerFast, BertModel 
from peft import PeftModel

dataset_name = "BGL" 
data_path = "/content/drive/MyDrive/LogLLM/BGL/bgl_test.csv"
Bert_path = "bert-base-uncased"
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_2"
configuration = "orginal"
pooling = "cls" 

max_content_len = 100 
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir = "/content/drive/MyDrive/LogLLM/results/" f"embeddings_{dataset_name}"
os.makedirs(output_dir, exist_ok=True) 
embedding_path = os.path.join( output_dir, f"{dataset_name}_embeddings_{configuration}.npy" ) 
metadata_path = os.path.join( output_dir, f"{dataset_name}_embedding_metadata_{configuration}.csv" )

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






