import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import apply_feature_engineering
from src.utils import get_logger, save_object
from src.config import DATASET_PATH, DATASET_SHEET, MODEL_PATH, PREPROCESSOR_PATH

logger = get_logger(__name__)

def train_model():
    """Treina o modelo de Random Forest e salva o modelo e o preprocessor."""
    logger.info("Iniciando o processo de treinamento...")
    
    # 1. Carregar e limpar dados
    logger.info(f"Lendo os dados de {DATASET_PATH}...")
    df = preprocess_pipeline(DATASET_PATH, DATASET_SHEET)
    logger.info(f"Dados carregados. Shape: {df.shape}")

    # 2. Engenharia de Features
    logger.info("Aplicando engenharia de features...")
    X_train, y_train, preprocessor = apply_feature_engineering(df, is_training=True)
    
    # 3. Treinamento do Modelo
    logger.info("Treinando RandomForestClassifier...")
    # Usando class_weight='balanced' pois as classes podem ser desbalanceadas (ex: Quartzo vs Topázio)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
    model.fit(X_train, y_train)
    
    # 4. Salvar os artefatos
    logger.info("Salvando modelo e preprocessor...")
    save_object(model, MODEL_PATH)
    save_object(preprocessor, PREPROCESSOR_PATH)
    
    logger.info("Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    train_model()
