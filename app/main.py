import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from app.schemas import PredictionRequest, PredictionResponse
from src.utils import load_object, get_logger
from src.config import MODEL_PATH, PREPROCESSOR_PATH
from src.feature_engineering import apply_feature_engineering

logger = get_logger(__name__)

app = FastAPI(title="Datathon Predictive Model API", description="API para classificar estudantes", version="1.0.0")

# Globais para o modelo e preprocessor
model = None
preprocessor = None

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo ao iniciar a API."""
    global model, preprocessor
    try:
        model = load_object(MODEL_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)
        logger.info("Modelo e preprocessor carregados com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar os artefatos: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API está rodando."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model, preprocessor
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Modelo não está carregado.")

    try:
        # Converter o request em um DataFrame de uma única linha
        # IMPORTANTE: As chaves do dicionário devem bater exatamente com as colunas que o preprocessor espera!
        data = {
            "Fase": [request.Fase],
            "Idade": [request.Idade],
            "Gênero": [request.Genero],
            "Ano ingresso": [request.Ano_ingresso],
            "Instituição de ensino": [request.Instituicao_de_ensino],
            "Fase Ideal": [request.Fase_Ideal],
            "Defasagem": [request.Defasagem]
        }
        
        df = pd.DataFrame(data)
        
        # Simples mock de Monitoramento (Data Drift local)
        logger.info(f"Monitoramento - Entrada do usuário: {data}")

        # Aplicar Feature Engineering
        X_transformed, _ = apply_feature_engineering(df, is_training=False, preprocessor=preprocessor)

        # Fazer predição
        pred = model.predict(X_transformed)
        
        return PredictionResponse(pedra_2024=pred[0])

    except Exception as e:
        logger.error(f"Erro na inferência: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
