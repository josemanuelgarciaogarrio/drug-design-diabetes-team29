import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, input_dim // 2)
        self.layer2 = nn.Linear(input_dim // 2, output_dim)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim_A, input_dim_B, backbone_output_dim, head_hidden_dim):
        super().__init__()
        self.backbone_A = nn.Sequential(
            nn.Linear(input_dim_A, input_dim_A // 2),
            nn.ReLU(),
            nn.Linear(input_dim_A // 2, backbone_output_dim)
        )
        self.backbone_B = nn.Sequential(
            nn.Linear(input_dim_B, input_dim_B // 2),
            nn.ReLU(),
            nn.Linear(input_dim_B // 2, backbone_output_dim)
        )
        self.head = nn.Sequential(
            # Input size es doble porque concatenamos ambas secuencias
            nn.Linear(backbone_output_dim * 2, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1) # Única salida para regresión
        )
    def forward(self, embeddingA, embeddingB):
        processed_A = self.backbone_A(embeddingA)
        processed_B = self.backbone_B(embeddingB)
        #Concatenamos a lo largo de la última dimensión
        fused_vector = torch.cat((processed_A, processed_B), dim=1)
        output = self.head(fused_vector)
        return output