## Alphafold predictor for diabetes purposes

A continuación se presenta la estructura del folder que estará destinado para realizar las predicciones mediante Docker. Como entrada se tendrá una cadena individual o un batch de cadenas, y como salida se tienen las métricas pLDDT e ipTM

```
alphafold_predictor/
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── esm_embedder.py
│   ├── mlp_predictor.py
│   └── batch_predictor.py
│
├── models/
│   └── mlp_weights.pth or final_model.joblib
│
├── output/              
│
└── run_prediction.py
```

El flujo normal de inferencia se presenta a continuación:
```
Usuario: "MKTLILAFLFASA"
         ↓
validate_sequence() → "MKTLILAFLFASA" ✓
         ↓
tokenizer → [0, 15, 18, 21, ...]
         ↓
ESM model → embedding [1280 valores]
         ↓
torch.tensor() → tensor([...])
         ↓
MLP model → prediction: 0.856
         ↓
resultado: {'sequence': '...', 'prediction': 0.856, ...}

```

