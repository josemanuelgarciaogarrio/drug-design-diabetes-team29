import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomProteinDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.sequences = [row.replace("/","") for row in df['seq'].tolist()] #Se concatena ambas cadenas como una sola
        plddt_scores = df['plddt'].astype(float)
        iptm_scores = df['i_ptm'].astype(float)
        self.plddts = torch.tensor(plddt_scores.values, dtype=torch.float32).view(-1,1)
        self.iptms = torch.tensor(iptm_scores.values, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        plddt_score = self.plddts[index]
        iptm_score = self.iptms[index]
        return sequence, plddt_score, iptm_score