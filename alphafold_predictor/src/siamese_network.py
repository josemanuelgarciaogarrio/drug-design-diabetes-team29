import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, encoding_dim):
        super().__init__()
        
        self.siamese_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # Añadimos Dropout para regularización
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, encoding_dim)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(encoding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Salida única para regresión
        )

    def forward(self, embeddingA, embeddingB):
        processed_A = self.siamese_tower(embeddingA)
        processed_B = self.siamese_tower(embeddingB)

        difference = torch.abs(processed_A - processed_B)
        combined_vector = torch.cat((processed_A, processed_B, difference), dim=1)

        prediction = self.regression_head(combined_vector)

        return prediction