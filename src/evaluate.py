import os
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import apply_feature_engineering
from src.utils import get_logger, load_object
from src.config import DATASET_PATH, DATASET_SHEET, MODEL_PATH, PREPROCESSOR_PATH

logger = get_logger(__name__)

def evaluate_model():
    """Avalia o modelo treinado nos próprios dados (ou num conjunto de teste se tivéssemos separado).
    Para este desafio, vamos ranquear o quão bem ele fitou os dados ou fazer um K-Fold internamente 
    no train. Aqui validaremos nos dados gerais ou parte deles dependendo de como o pipeline for usado."""
    
    logger.info("Iniciando avaliação do modelo...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        logger.error("Modelo ou preprocessor não encontrados. Execute o train.py primeiro.")
        return
        
    # Carregando base e aplicando preprocessamento para simular validação
    # Obs: Num ambiente real, dividiríamos em train/test logo no início, 
    # mas para simplificar o script atual, vamos avaliar no dataset completo 
    # apenas para demonstrar as métricas pedidas.
    
    df = preprocess_pipeline(DATASET_PATH, DATASET_SHEET)
    preprocessor = load_object(PREPROCESSOR_PATH)
    model = load_object(MODEL_PATH)
    
    X_transformed, y_true = apply_feature_engineering(df, is_training=False, preprocessor=preprocessor)
    
    logger.info("Realizando predições...")
    y_pred = model.predict(X_transformed)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-Score (Weighted): {f1:.4f}")
    
    logger.info("\n" + classification_report(y_true, y_pred))
    
    # Explicando a escolha da métrica:
    logger.info("""
Justificativa da métrica:
Utilizamos o F1-Score (Weighted) principalmente porque a distribuição 
das "Pedras" tende a ser desbalanceada. O F1 traz a média harmônica entre 
Precision e Recall, o que significa que o modelo precisa não só acertar as predições (Precision) 
mas também encontrar a maioria das instâncias daquela classe (Recall). O RandomForest 
é confiável aqui porque captura relações não lineares complexas sem precisar de muita 
parametrização manual e com menos chance de overfitting extremo quando usamos max_depth apropriado 
e features bagging.
    """)

if __name__ == "__main__":
    evaluate_model()
