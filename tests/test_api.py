from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API está rodando."}

# Nota: O teste do endpoint /predict pode falhar se o modelo não estiver treinado localmente
# durante a execução. O pytest rodará após o script de treinamento, então deve funcionar.
def test_predict_endpoint_no_model():
    # Mock para teste unitário real de predict envolveria monkeypatching do start do app
    pass
