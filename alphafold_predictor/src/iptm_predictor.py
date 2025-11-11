import xgboost as xgb
import logging
import torch
from src.siamese_network import SiameseNetwork
from src.config import MODEL_PATH_iptm_ml, MODEL_PATH_iptm_dl

logger = logging.getLogger(__name__)

class iptm_predictor:
    def __init__(self):
        self.model_ml = xgb.XGBRegressor()
        self.model_ml.load_model(MODEL_PATH_iptm_ml)

        self.model_dl = SiameseNetwork(embedding_dim=1280, hidden_dim=256, encoding_dim=64)
        self.model_dl.load_state_dict(torch.load(MODEL_PATH_iptm_dl))
        self.model_dl.eval()
        logger.info("Modelos cargados para predicción de iPTM")


    def get_features(self, emb1, emb2):
        #Pasamos por los features
        t1 = torch.tensor(emb1, dtype=torch.float32)
        t2 = torch.tensor(emb2, dtype=torch.float32)
        with torch.no_grad():
            encoding_a = self.model_dl.siamese_tower(t1.unsqueeze(0))
            encoding_b = self.model_dl.siamese_tower(t2.unsqueeze(0))
            difference = torch.abs(encoding_a - encoding_b)
            features = torch.cat((encoding_a, encoding_b, difference), dim=1)
            result = features.cpu().numpy()
        logger.info(f"Extracción de features usando SNN exitosa , dimensión de: {result.shape}")
        return result
    
    def get_features_batch(self, embs1, embs2):
        #Pasamos por los features
        t1 = torch.tensor(embs1, dtype=torch.float32)
        t2 = torch.tensor(embs2, dtype=torch.float32)
        with torch.no_grad():
            encoding_a = self.model_dl.siamese_tower(t1)
            encoding_b = self.model_dl.siamese_tower(t2)
            difference = torch.abs(encoding_a - encoding_b)
            features = torch.cat((encoding_a, encoding_b, difference), dim=1)
            result = features.cpu().numpy()
        logger.info(f"Extracción de features batch usando SNN exitosa , dimensión de: {result.shape}")
        return result

    def predict(self, data):
        preds = self.model_ml.predict(data)
        logger.info(f"Predicción de XGBoost con éxito")
        return preds

