import pandas as pd
import pytest
from src.feature_engineering import apply_feature_engineering

def test_apply_feature_engineering_training():
    df = pd.DataFrame({
        "Idade": [18, 20],
        "Defasagem": [-1, 0],
        "Gênero": ["M", "F"],
        "Instituição de ensino": ["Escola A", "Escola B"],
        "Fase": ["Fase 1", "Fase 2"],
        "Fase Ideal": ["Fase 1", "Fase 2"],
        "Ano ingresso": [2022, 2023],
        "Pedra 2024": ["Quartzo", "Agata"]
    })
    
    X_trans, y, preprocessor = apply_feature_engineering(df, is_training=True)
    
    assert X_trans is not None
    assert X_trans.shape[0] == 2
    assert y is not None
    assert list(y.values) == ["Quartzo", "Agata"]
    assert preprocessor is not None

def test_apply_feature_engineering_inference():
    # Cria df dummy sem alvo
    df_train = pd.DataFrame({
        "Idade": [18, 20],
        "Defasagem": [-1, 0],
        "Gênero": ["M", "F"],
        "Instituição de ensino": ["Escola A", "Escola B"],
        "Fase": ["Fase 1", "Fase 2"],
        "Fase Ideal": ["Fase 1", "Fase 2"],
        "Ano ingresso": [2022, 2023],
        "Pedra 2024": ["Quartzo", "Agata"]
    })
    
    _, _, preprocessor = apply_feature_engineering(df_train, is_training=True)
    
    # Base de inferência, sem a target "Pedra 2024"
    df_infer = df_train.drop(columns=["Pedra 2024"])
    
    X_trans, y = apply_feature_engineering(df_infer, is_training=False, preprocessor=preprocessor)
    
    assert X_trans is not None
    assert X_trans.shape[0] == 2
    assert y is None
