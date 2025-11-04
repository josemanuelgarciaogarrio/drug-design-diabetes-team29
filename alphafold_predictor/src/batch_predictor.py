import torch
import logging
from datetime import datetime
from src.esm_embedding import ESMEmbedder
from src.mlp_predictor import MLPPredictor
from src.config import MLP_WEIGHTS_PATH

logger = logging.getLogger(__name__)

class BatchPredictor:
    def __init__(self):
        logger.info("Inicializando BatchPredictor...")
        self.embedder = ESMEmbedder()
        logger.info(f"Cargando modelo MLP desde {MLP_WEIGHTS_PATH}")
        self.predictor = MLPPredictor()
        logger.info("BatchPredictor listo para predicciones")
    
    def predict_single(self, sequence):
        try:
            logger.info(f"Procesando predicción individual para secuencia: {sequence[:6]}...")
            #Paso 1. Generar Embedding con ESM
            embedding = self.embedder.get_embedding(sequence)
            #Paso 2. Hacer predicción con Modelo Ganador
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            prediction = self.predictor.predict(embedding_tensor)
            #Obtenemos predicción
            prediction_value = prediction.item() #Como solo es un valor, podemos usar item
            #Mostramos resultado
            result = {
                'sequence': sequence,
                'prediction': prediction_value,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"Predicción exitosa: {prediction_value:.4f}")
            
            return result
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            
            return {
                'sequence': sequence,
                'prediction': None,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }