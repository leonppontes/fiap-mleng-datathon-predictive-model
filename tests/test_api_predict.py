from fastapi.testclient import TestClient
import pytest
from app.main import app
from src.train import train_model

@pytest.fixture(autouse=True)
def ensure_model_is_loaded():
    # Treina/cria o arquivo se não existir, e chama o startup event pra carregar as globais
    import asyncio
    from app.main import startup_event
    from src.config import MODEL_PATH
    import os
    
    if not os.path.exists(MODEL_PATH):
        train_model()
    asyncio.run(startup_event())


client = TestClient(app)

def test_predict_endpoint_success():
    payload = {
        "Fase": "Fase 8",
        "Idade": 21,
        "Genero": "M",
        "Ano_ingresso": 2020,
        "Instituicao_de_ensino": "Escola X",
        "Fase_Ideal": "Fase 8",
        "Defasagem": 0
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "pedra_2024" in response.json()
    assert response.json()["versao_modelo"] == "1.0.0"

def test_predict_endpoint_invalid_data():
    payload = {
        "Fase": "Fase 8"
        # Faltando campos
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422 # Erro pydantic validation
