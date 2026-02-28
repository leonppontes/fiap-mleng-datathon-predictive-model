import pandas as pd
import numpy as np
from src.config import FEATURES_ALL, TARGET

def load_data(filepath: str, sheet_name: str) -> pd.DataFrame:
    """Carrega a base de dados."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra as colunas de interesse, remove nulos ou preenche adequadamente."""
    # Filtrar apenas as colunas que importam para o modelo.
    cols_to_keep = FEATURES_ALL + [TARGET]
    
    # Nem todas as colunas podem existir e podem ter pequenos typos nos nomes. 
    # Validamos as colunas que existem no dataframe
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].copy()
    
    # Remover registros que não possuem a variável resposta (Target) preenchida
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        
    # Filtrar o Target apenas para os valores válidos (Quartzo, Agata, Ametista, Topázio)
    valid_targets = ['Quartzo', 'Agata', 'Ametista', 'Topázio']
    if TARGET in df.columns:
        df = df[df[TARGET].isin(valid_targets)]
        
    # Preencher nulos nas features numéricas com a mediana, e categóricas com "Desconhecido"
    for col in df.columns:
        if col == TARGET:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Desconhecido")
            
    return df

def preprocess_pipeline(filepath: str, sheet_name: str) -> pd.DataFrame:
    """Wrapper para ler e limpar a base de dados."""
    df = load_data(filepath, sheet_name)
    df_clean = clean_data(df)
    return df_clean
