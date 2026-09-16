import os
import re
from pathlib import Path
import numpy as np
import pandas as pd #
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LogLLM
#from sentencebert_model import LogLLM
#from bge_model import LogLLM
#from qwen_model import LogLLM
from customDataset import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
batch_size = 32
dataset_name = 'BGL'   # 'Thunderbird' 'HDFS_v1'  'BGL'  'Liberty'
data_path = "/content/drive/MyDrive/LogLLM/BGL/bgl_test.csv"
#data_path = "/content/drive/MyDrive/LogLLM/Thunderbird/test.csv"
#output_path = "/content/drive/MyDrive/LogLLM/BGL/test_preds_original.csv"
output_path = "/content/drive/MyDrive/LogLLM/BGL/test_preds_reproduced.csv"

Bert_path = "bert-base-uncased"
#Bert_path = "sentence-transformers/all-mpnet-base-v2"
#Bert_path = "BAAI/bge-m3"
#Bert_path = "Qwen/Qwen3-Embedding-0.6B"
Llama_path = "meta-llama/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
#ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}"
ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_2"
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_meanpool"
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_mpnet"
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_mpnet_finetune"
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_bgem3cls"
#ft_path = f"/content/drive/MyDrive/LogLLM/results/ft_model_{dataset_name}_bgem3mean"

device = torch.device("cuda:0")

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

print(
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'device: {device}')

def evalModel(model, dataloader):
    model.eval()

    preds = []

    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs,seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:
                match = re.search(r'normal|anomalous', text, re.IGNORECASE)
                if match:
                    preds.append(match.group())
                else:
                    print(f'error :{text}')
                    preds.append('')

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    gt = dataloader.dataset.get_label()

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    num_anomalous = (gt == 1).sum()
    num_normal = (gt == 0).sum()

    print(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}')

    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal =  (preds == 0).sum()

    print(
        f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}')

    print(f'precision: {precision}, recall: {recall}, f1: {f}, acc: {acc}')

    return preds #

if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    #evalModel(model, dataloader)
    preds = evalModel(model, dataloader) #

    # NEW: Save predections to a new CSV
    test_df = pd.read_csv(data_path)

    test_df['Preprocessed_Content'] = test_df['Content'].apply(
        lambda x: ' ;-; '.join(
            replace_patterns(log)
            for log in str(x).split(' ;-; ')
        )
    )

    assert len(preds) == len(test_df)

    test_df['Predicted_Label'] = preds 

    test_df.to_csv(output_path, index=False) 
    print(f'Predictions saved to: {output_path}')
