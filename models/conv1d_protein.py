import torch
import torch.nn as nn
import torch.nn.functional as F 


class Simple1CNN(nn.Module):
    def __init__(self, embedding_dim=1280, num_filters=256, kernel_size=5, output_dim=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.pool= nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, embeddings_residuo):
        x = embeddings_residuo.permute(0, 2,1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        output = self.fc(x)
        return output
