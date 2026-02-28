from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    Fase: str
    Idade: int
    Genero: str
    Ano_ingresso: int
    Instituicao_de_ensino: str
    Fase_Ideal: str
    Defasagem: int

class PredictionResponse(BaseModel):
    pedra_2024: str
    versao_modelo: str = "1.0.0"
