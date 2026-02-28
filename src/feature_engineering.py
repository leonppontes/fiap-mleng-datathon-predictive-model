import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.config import FEATURES_NUMERIC, FEATURES_CATEGORICAL

def create_preprocessor() -> ColumnTransformer:
    """
    Cria a pipeline do scikit-learn para engenharia de atributos:
    - Numéricos: Imputação com mediana e Standard Scaling.
    - Categóricos: Imputação com string e OneHotEncoding (ou OrdinalEncoding dependendo do tipo da feature).
    """

    # Transformador para variáveis numéricas + 'Ano ingresso' (tratado numericamente aqui)
    numeric_features = FEATURES_NUMERIC + ["Ano ingresso"]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = FEATURES_CATEGORICAL
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def apply_feature_engineering(df: pd.DataFrame, is_training: bool = True, preprocessor=None):
    """Aplica o preprocessor aos dados."""
    target_col = 'Pedra 2024'
    
    # Separa X e y se a coluna target existir (Treinamento)
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    if is_training:
        preprocessor = create_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        return X_transformed, y, preprocessor
    else:
        if preprocessor is None:
            raise ValueError("O preprocessor treinado deve ser passado para dados de teste/inferência.")
        X_transformed = preprocessor.transform(X)
        return X_transformed, y
