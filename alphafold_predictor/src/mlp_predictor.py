import torch.nn as nn
import torch
import logging
from src.config import MLP_WEIGHTS_PATH, DEVICE

logger = logging.getLogger(__name__)

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

class MLPPredictor:
    def __init__(self):
        logger.info(f"Cargando modelo MLP completo desde: {MLP_WEIGHTS_PATH}")
        try:
            self.model = MLP_pLDDT(input_size=1280, hidden_sizes=[1280,512,254], output_size=1)
            # Cargar pesos
            self.model.load_state_dict(
                torch.load(MLP_WEIGHTS_PATH, map_location=DEVICE)
            )
            self.model.eval()
            self.model.to(DEVICE)
            logger.info(f"Modelo MLP cargado exitosamente en {DEVICE}")
            logger.info(f"Arquitectura del modelo: {self.model}")
        except Exception as e:
            logger.error(f"Error al cargar modelo MLP: {str(e)}")
            raise

    def predict(self, embeddings):
        self.model.eval()
        with torch.no_grad():
            embeddings = embeddings.to(DEVICE)
            predictions = self.model(embeddings)
        return predictions
    
    def predict_batch(self, embeddings_batch):
        return self.predict(embeddings_batch)
