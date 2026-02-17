import os
from pathlib import Path
import torch
from tqdm import tqdm
from torch import nn, optim
from model import LogLLM
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler

device = torch.device("cuda:0")

# Hyperparameters
