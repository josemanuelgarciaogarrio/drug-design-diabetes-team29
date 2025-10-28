import torch.nn as nn

class MLP_pLDDT(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, embedding_A):
        #x = torch.cat((embedding_A, embedding_B), dim=1)
        output = self.network(embedding_A)
        return output

#input_size = 1024
#hidden_sizes = [1024,512,254]  # 3 capas ocultas
#output_size = 1

#model = MLP_pLDDT(input_size, hidden_sizes, output_size)
#print(model)

#criterion = nn.MSELoss()  # Mean Squared Error para regresión
#optimizer = optim.Adam(model.parameters(), lr=0.001)


