import os
import pytest
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import MODEL_PATH, PREPROCESSOR_PATH

def test_train_model():
    # Treinar modelo (pode demorar pouco com apenas RF simples local)
    train_model()
    
    # Verifica se os arquivos foram criados
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(PREPROCESSOR_PATH)

def test_evaluate_model(caplog):
    # O caplog pega as mensagens do logger default para testes unitários
    evaluate_model()
    
    assert "Accuracy:" in caplog.text
    assert "F1-Score" in caplog.text
