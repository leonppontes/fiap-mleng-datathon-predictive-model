import os

# Project root based on this file's location (src/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# We expect the dataset to be placed in the data directory
DATASET_NAME = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)
DATASET_SHEET = "PEDE2024"

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

# Columns to use
FEATURES_NUMERIC = ["Idade", "Defasagem"]
# "Fase" and "Fase Ideal" are technically categorical/ordinal.
# "Ano ingresso" is discrete/categorical/numerical (we'll treat as numerical or categorical)
FEATURES_CATEGORICAL = ["Gênero", "Instituição de ensino", "Fase", "Fase Ideal"]

FEATURES_ALL = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["Ano ingresso"]

TARGET = "Pedra 2024"
TARGET_CLASSES = ["Quartzo", "Agata", "Ametista", "Topázio"]
