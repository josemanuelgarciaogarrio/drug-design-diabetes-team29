import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface")

MODEL_PATH = MODEL_DIR / "best_model_plddt_vf.joblib"
MODEL_PATH_iptm_dl = MODEL_DIR / "best_model_snn_vf.pth"
MODEL_PATH_iptm_ml = MODEL_DIR / "mi_modelo_ganador_xgb_vf.json"


BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))

DEVICE = "cpu"

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

OUTPUT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_OUTPUT_FILENAME = "predictions.csv"