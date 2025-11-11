import joblib
from src.config import MODEL_PATH

class pLDDT_predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
    def predict(self, data, batch=False):
        if batch:
            preds = self.model.predict(data)
        else:
            preds = self.model.predict(data.reshape(1,-1))
        return preds

