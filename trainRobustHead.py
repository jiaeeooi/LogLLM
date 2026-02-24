import os
from pathlib import Path
import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from newModel import LogLLM
from robustDataset import RobustDataset, RobustCollator, BalancedSampler

# ===============================
# Hyperparameters
# ===============================

dataset_name = 'BGL'
batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size

n_epochs_robust = 2
lr_robust = 1e-4

max_content_len = 100
max_seq_len = 128
min_less_portion = 0.3

data_path = r'/content/data/{}/train.csv'.format(dataset_name)
Bert_path = "bert-base-uncased"
Llama_path = "meta-llama/Meta-Llama-3-8B"

device = torch.device("cuda:0")

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
robust_path = os.path.join(ft_path, 'robust.pt')




print(f'n_epochs_robust: {n_epochs_robust}\n'
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'micro_batch_size: {micro_batch_size}\n'
f'lr_robust: {lr_robust}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'min_less_portion: {min_less_portion}\n'
f'device: {device}')


# ===============================
# Loss
# ===============================

def invariance_loss(z1, z2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return 1 - (z1 * z2).sum(dim=1).mean()


# ===============================
# Utility
# ===============================

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()

    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")


# ===============================
# Robust Training
# ===============================

def trainRobustHead(model, dataloader, gradient_accumulation_steps, n_epochs, lr):

    print_number_of_trainable_model_parameters(model)

    optimizer = torch.optim.AdamW(
        model.robust_head.parameters(),  #
        lr=lr
    )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f'scheduler_step: {scheduler_step}')

    steps = 0

    for epoch in range(n_epochs):
        total_loss, total_count = 0, 0  #
        
        pbar = tqdm(dataloader, desc=f'Robust Epoch {epoch+1}/{n_epochs}')  #

        for i_th, batch_i in enumerate(pbar):
            steps += 1

            inputs = batch_i['inputs'].to(device)
            para_inputs = batch_i['para_inputs'].to(device)
            #inputs = {k: v.to(device) for k, v in batch_i['inputs'].items()}
            #para_inputs = {k: v.to(device) for k, v in batch_i['para_inputs'].items()}

            # Forward
            z_orig = model.encode_with_robust_head(inputs)
            z_para = model.encode_with_robust_head(para_inputs)

            loss = invariance_loss(z_orig, z_para)
            loss = loss / gradient_accumulation_steps

            loss.backward()

            # Gradient accumulation
            if ((i_th + 1) % gradient_accumulation_steps == 0) or \
               ((i_th + 1) == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps * inputs.size(0)
            total_count += inputs.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()

            pbar.set_postfix(
                lr=scheduler.get_last_lr()[0],
                loss=loss.item() * gradient_accumulation_steps
            )

        if total_count > 0:
            train_loss_epoch = total_loss / total_count
            print(f"[Robust Epoch {epoch+1}/{n_epochs}] "
                  f"[loss: {train_loss_epoch:.6f}]")


# ===============================
# Main
# ===============================

if __name__ == '__main__':

    print(f'dataset: {data_path}')

    dataset = RobustDataset(data_path, drop_duplicates=False)

    model = LogLLM(
        Bert_path,
        Llama_path,
        ft_path=ft_path,
        device=device,
        max_content_len=max_content_len,
        max_seq_len=max_seq_len
    )

    tokenizer = model.Bert_tokenizer
    collator = RobustCollator(
        tokenizer,
        max_seq_len=max_seq_len,
        max_content_len=max_content_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=4,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion),
        collate_fn=collator,
        drop_last=True
    )

    # Freeze everything except robust head
    print("*" * 10 + "Start training Robust Head" + "*" * 10)
    model.set_train_only_robust_head()
    model.train()  #

    trainRobustHead(
        model,
        dataloader,
        gradient_accumulation_steps,
        n_epochs_robust,
        lr_robust
    )

    # Save only robust head
    torch.save(model.robust_head.state_dict(), robust_path)  #
    print(f"Robust head saved to {robust_path}")