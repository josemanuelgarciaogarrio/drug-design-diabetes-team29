import torch
import sys
import logging
from datetime import datetime
from src.esm_embedding import ESMEmbedder
from src.plddt_predictor import pLDDT_predictor
from src.iptm_predictor import iptm_predictor
from src.custom_dataloader import CustomProteinDataset
from src.config import MODEL_PATH, MODEL_PATH_iptm_dl, MODEL_PATH_iptm_ml
from src.create_output_file import create_output_file

logger = logging.getLogger(__name__)

class BatchPredictor:
    def __init__(self):
        logger.info("Inicializando BatchPredictor...")
        self.embedder = ESMEmbedder()
        logger.info(f"Cargando modelo GradientBoost desde {MODEL_PATH}")
        self.predictor = pLDDT_predictor()
        logger.info(f"Cargando modelos Snn y XGboost desde {MODEL_PATH_iptm_dl} y {MODEL_PATH_iptm_ml}")
        self.predictor_iptm = iptm_predictor()
        logger.info("BatchPredictor listo para predicciones")
    
    def predict_single(self, sequence):
        try:
            logger.info(f"Procesando predicción individual para secuencia: {sequence[:6]}...")
            #Paso 1. Generar Embedding con ESM
            embedding = self.embedder.get_embedding(sequence)
            #Paso 2. Hacer predicción con Modelo Ganador para PLDDT
            prediction = self.predictor.predict(embedding).item()
            #Paso 3. Predicción para iPTM, primero obtenemos el embedding
            embedding_iptm1,embedding_iptm2 = self.embedder.get_embedding_iptm(sequence)
            #Paso 4. Predicción para iPTM, extraemos features de las dos cadenas de embeddings previas
            features_iptm = self.predictor_iptm.get_features(embedding_iptm1, embedding_iptm2)
            #Paso 5. Predicción para iPTM, pasamos los features al XGBoost
            predidction_iptm = self.predictor_iptm.predict(features_iptm).item()
            #Mostramos resultado
            result = {
                'sequence': sequence,
                'prediction_pLDDT': prediction,
                'prediction_iPTM': predidction_iptm,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            logger.info(f"Predicción exitosa pLDDT: {prediction:.4f}")  
            logger.info(f"Predicción exitosa iPTM: {predidction_iptm:.4f}")   
            return result
        
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            logger.error(f"Detalles: {embedding}")
            logger.error(f"Detalles: {prediction}")

            return {
                'sequence': sequence,
                'prediction': None,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    def predict_batch(self, input_file):
        try:
            logger.info(f"Procesando predicción individual para input en: {input_file}")
            #Paso 0. Leemos archivo y generamos dataloader
            dataset = CustomProteinDataset(input_file)
            logger.info(f"Registros de dataset: {len(dataset)}")
            #Paso 1. Generar Embedding con ESM
            embeddings = self.embedder.get_embedding_batch(dataset)
            logger.info(f"Registros después del embedding: {embeddings.shape}")
            #Paso 2. Hacer predicción con Modelo Ganador
            predictions = self.predictor.predict(embeddings, batch=True)
            #Paso 3. Predicción para iPTM, primero obtenemos el embedding
            embeddings_iptm1,embeddings_iptm2 = self.embedder.get_embedding_batch_iptm(dataset)
            #Paso 4. Predicción para iPTM, extraemos features de las dos cadenas de embeddings previas
            features_iptm = self.predictor_iptm.get_features_batch(embeddings_iptm1, embeddings_iptm2)
            #Paso 5. Predicción para iPTM, pasamos los features al XGBoost
            predidction_iptms = self.predictor_iptm.predict(features_iptm)
            #Paso 6. Escribir en archivo
            create_output_file(dataset=dataset, preds_plddt=predictions, preds_iptm=predidction_iptms)
            #Mostramos resultado
            result = {
                'sequence': 'batch',
                'prediction_pLDDT': predictions,
                'prediction_iPTM':predidction_iptms,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }

            logger.info(f"Predicción exitosa por batches pLDDT")
            logger.info(f"Predicción exitosa por batches iPTM")   
            return result
        
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return {
                'sequence': None,
                'prediction': None,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }