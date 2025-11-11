import torch
from transformers import AutoTokenizer, AutoModel
import logging
import numpy as np
from src.config import ESM_MODEL_NAME, DEVICE, MAX_SEQ_LENGTH, VALID_AMINO_ACIDS

logger = logging.getLogger(__name__)

class ESMEmbedder:
    def __init__(self):
        logger.info(f"Cargando modelo ESM: {ESM_MODEL_NAME}")
        torch.manual_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.model = AutoModel.from_pretrained(ESM_MODEL_NAME)
        self.model.eval()
        self.model.to(DEVICE)
        logger.info("Modelo ESM cargado exitosamente")

    def validate_sequence(self, sequence:str):
        sequence = sequence.upper().strip().replace("/","")

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
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding_por_residuo = outputs.last_hidden_state
            embedding_por_secuencia = embedding_por_residuo.mean(dim=1) 
        
        embedding = embedding_por_secuencia.numpy()[0]
        logger.info(f"Embedding generado con dimensión: {embedding.shape}")
        return embedding
    

    def get_embedding_batch(self, seqs):
        data_seqs = []
        for seq, plddt, iptm, seq1, seq2 in seqs:
            data_seq = self.get_embedding(seq)
            data_seqs.append(data_seq)
        data_seqs_np = np.array(data_seqs)
        logger.info(f"Embeddings creados con dimensiones correctamente")
        return data_seqs_np

    def get_embedding_iptm(self, seq):
        seq1 = seq.split("/")[0]
        seq2 = seq.split("/")[1]
        emb1 = self.get_embedding(seq1)
        emb2 = self.get_embedding(seq2)
        logger.info(f"Embeddings creados para iptm correctamente")
        return (emb1, emb2)
    
    def get_embedding_batch_iptm(self, seqs):
        data_seqs1 = []
        data_seqs2 = []
        for seq, plddt, iptm, seq1, seq2 in seqs:
            data_seq1 = self.get_embedding(seq1)
            data_seq2 = self.get_embedding(seq2)
            data_seqs1.append(data_seq1)
            data_seqs2.append(data_seq2)
        data_seqs_np1 = np.array(data_seqs1)
        data_seqs_np2 = np.array(data_seqs2)
        logger.info(f"Embeddings para iptm creados correctamente: {data_seqs_np1.shape}, {data_seqs_np2.shape}")
        return data_seqs_np1,data_seqs_np2