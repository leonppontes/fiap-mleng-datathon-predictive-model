import pandas as pd
import numpy as np
import pytest
from src.preprocessing import clean_data

def test_clean_data_handles_missing_target():
    # Cria df dummy sem alvo 
    df = pd.DataFrame({
        "Idade": [20, 21],
        "Pedra 2024": ["Quartzo", np.nan]
    })
    
    df_clean = clean_data(df)
    
    # Deve dropar o registro com NaN na coluna target
    assert len(df_clean) == 1
    assert "Quartzo" in df_clean["Pedra 2024"].values

def test_clean_data_imputes_numerical():
    df = pd.DataFrame({
        "Idade": [20, np.nan, 22],
        "Pedra 2024": ["Quartzo", "Agata", "Ametista"]
    })
    
    df_clean = clean_data(df)
    # Mediana de 20 e 22 é 21
    assert df_clean["Idade"].iloc[1] == 21.0

def test_clean_data_filters_invalid_targets():
    df = pd.DataFrame({
        "Idade": [20, 21, 22],
        "Pedra 2024": ["Quartzo", "Invalido", "Agata"]
    })
    
    df_clean = clean_data(df)
    # Apenas Quartzo e Agata são mantidos
    assert len(df_clean) == 2
    assert "Invalido" not in df_clean["Pedra 2024"].values
