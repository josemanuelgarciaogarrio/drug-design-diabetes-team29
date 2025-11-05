import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class CustomProteinDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.sequences = [row.replace("/","") for row in df['seq'].tolist()] #Se concatena ambas cadenas como una sola
        plddt_scores = df['plddt'].astype(float)
        self.plddts = torch.tensor(plddt_scores.values, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        plddt_score = self.plddts[index]
        return sequence, plddt_score