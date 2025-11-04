import torch
from transformers import AutoTokenizer, AutoModel
import logging
from src.config import ESM_MODEL_NAME, DEVICE, MAX_SEQ_LENGTH, VALID_AMINO_ACIDS

logger = logging.getLogger(__name__)

class ESMEmbedder:
    def __init__(self):
        logger.info(f"Cargando modelo ESM: {ESM_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.model = AutoModel.from_pretrained(ESM_MODEL_NAME)
        self.model.eval()
        self.model.to(DEVICE)
        logger.info("Modelo ESM cargado exitosamente")

    def validate_sequence(self, sequence:str):
        sequence = sequence.upper().strip()
        if not sequence:
            raise ValueError("La secuencia no debe estar vacía")
        if len(sequence) > MAX_SEQ_LENGTH:
            raise ValueError(
                f"Secuencia demasiada larga: {len(sequence)}"
                f"(máximo: {MAX_SEQ_LENGTH})"
            )
        invalid_chars = set(sequence) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Caracteres invalidos en secuencia: {invalid_chars}"
                f"Solo se permiten aminoacidos estandar: {VALID_AMINO_ACIDS}"
            )
        return sequence
    
    def get_embedding(self, sequence):
        sequence = self.validate_sequence(sequence)
        logger.info(f"Generando embedding para secuencia de longitud {len(sequence)}")

        inputs = self.tokenizer(
            sequence, 
            padding=True, #Padding para el mismo tamaño
            truncation=True, #Recorta en caso de que sean muy largas
            return_tensors="pt", #Regresa tensores de pytorch
            max_length = MAX_SEQ_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1)
        logger.info(f"Embedding generado con dimensión: {embedding.shape}")
        return embedding