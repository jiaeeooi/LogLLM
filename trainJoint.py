import os
from pathlib import Path
import torch
from tqdm import tqdm
from torch import optim
from torch import nn
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

n_epochs_robust = 1
lr_robust = 1e-4

max_content_len = 100
max_seq_len = 128
min_less_portion = 0.3

data_path = "/content/bgl_train_paraphrased.csv"
Bert_path = "bert-base-uncased"
Llama_path = "meta-llama/Meta-Llama-3-8B"

device = torch.device("cuda:0")

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
robust_path = os.path.join(ft_path, 'robust.pt')
new_projector_path = os.path.join(ft_path, 'newprojector.pt')




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
# SimCLR Loss
# ===============================

def simclr_loss(z1, z2, temperature=0.07):

    B = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    positives = torch.cat([
        torch.arange(B, 2*B),
        torch.arange(0, B)
    ]).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss

# ===============================
# VICReg Loss
# ===============================

def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    """
    VICReg loss:
    - sim_coeff: invariance weight
    - std_coeff: variance weight
    - cov_coeff: covariance weight
    """
    # Invariance loss
    invariance_loss = F.mse_loss(z1, z2) 

    # Variance loss
    def variance_loss(z):  
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1 - std))
    var_loss = variance_loss(z1) + variance_loss(z2)

    # Covariance loss
    def covariance_loss(z):
        B, D = z.size()
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D
    cov_loss = covariance_loss(z1) + covariance_loss(z2)

    # Final weighted sum
    loss = (
        sim_coeff * invariance_loss
        + std_coeff * var_loss
        + cov_coeff * cov_loss
    )
    return loss


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

def trainJoint(model, dataloader, gradient_accumulation_steps, n_epochs, lr):

    print_number_of_trainable_model_parameters(model)

    optimizer = torch.optim.AdamW(
        list(model.robust_head.parameters()) + list(model.projector.parameters()),  #
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

        optimizer.zero_grad()

        for i_th, batch_i in enumerate(pbar):
            #steps += 1

            #inputs = batch_i['inputs'].to(device)
            #para_inputs = batch_i['para_inputs'].to(device)
            inputs = {k: v.to(device) for k, v in batch_i['inputs'].items()}
            para_inputs = {k: v.to(device) for k, v in batch_i['para_inputs'].items()}
            seq_positions = batch_i['seq_positions']
            labels = batch_i['labels']

            with torch.no_grad:
                h_orig = model.Bert_model(**inputs).pooler_output.float()
                h_para = model.Bert_model(**para_inputs).pooler_output.float()
                #h_orig = model.Bert_model(**inputs).last_hidden_state[:, 0].float()
                #h_para = model.Bert_model(**para_inputs).last_hidden_state[:, 0].float()

            # Forward
            z_orig = model.robust_head(h_orig)
            z_para = model.robust_head(h_para)

            loss_vicreg = vicreg_loss(z_orig, z_para) 
            
            labels_np = labels.clone().cpu().numpy().astype(object)
            labels_np[labels_np == 0] = 'normal'
            labels_np[labels_np == 1] = 'anomalous'

            logits, targets = model.train_helper(inputs, seq_positions, labels_np)

            #criterion = nn.CrossEntropyLoss(reduction='mean')
            #loss_task = criterion(logits, targets)
            loss_task = F.cross_entropy(logits, targets)

            lambda_vicreg = 0.05 # Tune
            loss = (loss_task + lambda_vicreg * loss_vicreg) / gradient_accumulation_steps

            loss.backward()

            # Gradient accumulation
            if ((i_th + 1) % gradient_accumulation_steps == 0) or \
               ((i_th + 1) == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

                if steps % scheduler_step == 0:
                    scheduler.step()

            #total_loss += loss.item() * gradient_accumulation_steps * inputs.size(0)
            #total_count += inputs.size(0)
            batch_size_actual = next(iter(inputs.values())).size(0)
            total_loss += loss.item() * gradient_accumulation_steps * batch_size_actual
            total_count += batch_size_actual

            #if steps % scheduler_step == 0:
                #scheduler.step()

            #pbar.set_postfix(
                #lr=scheduler.get_last_lr()[0],
                #loss=loss.item() * gradient_accumulation_steps
            #)
            pbar.set_postfix(
                lr=optimizer.param_groups[0]['lr'],
                loss=loss.item() * gradient_accumulation_steps,
                task=loss_task.item(),
                vicreg=loss_vicreg.item()
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
    for p in model.projector.parameters():
        p.requires_grad = True

    model.train()  #

    trainJoint(
        model,
        dataloader,
        gradient_accumulation_steps,
        n_epochs_robust,
        lr_robust
    )

    # Save model
    torch.save(model.robust_head.state_dict(), robust_path)  #
    print(f"Robust head saved to {robust_path}")

    torch.save(model.projector.state_dict(), new_projector_path)
    print(f"Projector saved to {_path}")