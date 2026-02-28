# Visão Geral do Projeto

**Objetivo:** Este projeto resolve o desafio de construir uma pipeline completa de Machine Learning para prever a classificação "Pedra 2024" (Quartzo, Agata, Ametista, Topázio) de estudantes baseado em atributos demográficos (Idade, Gênero) e atributos acadêmicos (Fase, Ano Ingresso, etc.).

**Solução Proposta:** Construção de uma pipeline em Python utilizando `scikit-learn` para processamento e treinamento de um modelo *Random Forest*, juntamente com o desenvolvimento de uma API via `FastAPI` para deploy e predições em tempo real. Toda a aplicação foi empacotada com o Docker para garantir que pudesse ser executada com o ambiente de maneira isolada. Testes unitários com o `pytest` alcançaram >90% de code coverage em todo o projeto.

**Stack Tecnológica:**
- **Linguagem:** Python 3.12
- **Frameworks de ML:** `scikit-learn`, `pandas`, `numpy`
- **API:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Serialização:** `joblib`
- **Testes:** `pytest`, `pytest-cov`
- **Empacotamento:** `Docker`
- **Monitoramento:** Logging básico estruturado para simulação de detecção de drift de dados.

# Estrutura do Projeto (Diretórios e Arquivos)

```bash
📦 fiap-mleng-datathon-predictive-model
 ┣ 📂 app
 ┃ ┣ 📜 main.py                  # Ponto de entrada da API FastAPI
 ┃ ┣ 📜 schemas.py               # Modelos Pydantic (Request/Response)
 ┣ 📂 data
 ┃ ┣ 📜 BASE DE DADOS...xlsx     # Base de dados (deve conter a aba PEDE2024)
 ┣ 📂 models
 ┃ ┣ 📜 preprocessor.joblib      # Artefato da pipeline de transformação (gerado)
 ┃ ┣ 📜 random_forest_model.joblib # Modelo Random Forest treinado (gerado)
 ┣ 📂 src
 ┃ ┣ 📜 config.py                # Caminhos e constantes do sistema
 ┃ ┣ 📜 evaluate.py              # Script para avaliação de métricas 
 ┃ ┣ 📜 feature_engineering.py   # Transformadores categóricos e numéricos
 ┃ ┣ 📜 preprocessing.py         # Tratamento de nulos e seleção de dados
 ┃ ┣ 📜 train.py                 # Script orquestrador de treinamento
 ┃ ┣ 📜 utils.py                 # Fofocas de IO e loggings
 ┣ 📂 tests
 ┃ ┣ 📜 test_api.py              # Testes do endpoint de healthcheck
 ┃ ┣ 📜 test_api_predict.py      # Testes do endpoint preditor
 ┃ ┣ 📜 test_feature_engineering.py # Testes dos artefatos de transform
 ┃ ┣ 📜 test_preprocessing.py    # Teste de limpeza de dados
 ┃ ┣ 📜 test_train_evaluate.py   # Testes dos scripts de ML
 ┣ 📜 Dockerfile                 # Imagem containerizada do ambiente
 ┣ 📜 README.md                  # Este arquivo
 ┣ 📜 requirements.txt           # Dependências pip
```

# Instruções de Deploy

## Apenas API Local (Sem Docker)

1. Garanta ter o Python 3.12+ em sua máquina.
2. Instale as dependências:
   `pip install -r requirements.txt`
3. Treine o modelo para gerar os arquivos passivos em `/models` antes de testar a base:
   `python src/train.py`
   *(Nota: O script `train.py` é o orquestrador principal. Ele **automaticamente** aciona os módulos `preprocessing.py` e `feature_engineering.py` para limpar a base de dados original de Excel e codificar as features antes do treinamento).*
   
4. Avalie as métricas:
   `python src/evaluate.py`
5. Rodado a API local:
   `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
6. Rode os testes rodando a bateria com coverage:
   Para que o Python saiba a raiz do pacote dos testes, insira o `PYTHONPATH` antes de rodar o comando:
   No **Windows PowerShell**:
   ```powershell
   $env:PYTHONPATH="."; python -m pytest --cov=src --cov=app tests/
   ```
   No **Linux/Mac**:
   ```bash
   PYTHONPATH="." python -m pytest --cov=src --cov=app tests/
   ```

## Ambiente Containerizado (Docker - Recomendado)

Para subir o sistema isolado em um container:
1. Certifique-se de que treinou o modelo localmente via `python src/train.py` para não faltar os arquivos `.joblib` em `./models/` na hora de exportar a imagem. 
2. Construa a Imagem Docker (na pasta raiz):
   ```bash
   docker build -t datathon-model:latest .
   ```
3. Execute o Container:
   ```bash
   docker run -p 8000:8000 datathon-model:latest
   ```
   A aplicação subirá no servidor uvicorn apontando para a porta 8000 na sua máquina host.

# Exemplos de Chamadas à API

A API expõe o endpoint de predições `/predict`. Abaixo exemplos utilizando cURL para validar:

### Health Check 

```bash
curl -X GET http://localhost:8000/
```
**Resposta:**
```json
{
  "status": "ok",
  "message": "API está rodando."
}
```

### Predict 

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
            "Fase": "Fase 8",
            "Idade": 21,
            "Genero": "M",
            "Ano_ingresso": 2021,
            "Instituicao_de_ensino": "VAGA MACRO",
            "Fase_Ideal": "Fase 8",
            "Defasagem": 0
         }'
```
**Resposta Esperada:**
```json
{
  "pedra_2024": "Quartzo",
  "versao_modelo": "1.0.0"
}
```

# Etapas do Pipeline de Machine Learning

Foram codificadas de forma modularizada no diretório `src/`:

1. **Pré-processamento dos Dados (`preprocessing.py`):**
   Limpeza básica como dropar dados com variavel target faltante (Pedra 2024=nulo), filtra linhas cujos alvos contêm descrições inválidas sobre os quatro grupos esperados. Preenchimento de dados numéricos omissos com mediana da própria coluna ou strings categóricas com `"Desconhecido"`.

2. **Engenharia de Features (`feature_engineering.py`):**
   Uso do `ColumnTransformer` (sci-kit learn). A codificação divide atributos:
   - Numéricos e "Ano ingresso": Escalamento via `StandardScaler` e imputação com Mediana.
   - Categóricos/Ordinais (`Gênero`, `Instituição`, `Fase`): Imputação constante para omissos e One-Hot Encoder para expansão categórica vetorial livre de suposição de grandeza. O Transformador compõe a "pipeline" completa que depois é unida ao exportável (`joblib`).

3. **Treinamento e Validação (`train.py` && `evaluate.py`):**
   Treinamento utiliza modelo Random Forest com `class_weight='balanced'` por conta nativa do banco de dados ser desequilibrado nos 4 rótulos (Há massivamente mais classificação de grupo "Quartzo"). É utilizado RandomForest por capturar fronteiras não lineares sem precisar de otimizações de gradiente robustas ou customizações profundas para obter ótimos scores na modelagem.

4. **Persistência (`utils.py`):**
   O pré-processamento treinado e as matrizes internas de florestas são salvas para serem acopladas e enviadas à API sem vazar dados entre as partições.

5. **Monitoramento Local (Simulação de Data Drift):**
   No ambiente atual, implementamos a fundação para o monitoramento de modelos em produção utilizando **Logs Estruturados**. 
   - **Como funciona hoje:** Toda vez que a API recebe dados para predição, o módulo `app/main.py` registra (log) as características do aluno consultado diretamente na saída do terminal/container.
   - **Evolução para Produção:** Num cenário real cloud, esses logs de inferência seriam capturados automaticamente por ferramentas de observabilidade (ex: Datadog, AWS CloudWatch ou a stack ELK - Elasticsearch, Logstash, Kibana). Com os dados centralizados nesses observadores, criaríamos *Dashboards* para comparar a distribuição de atributos (ex: Idade, Fase) enviados pelos usuários contra a distribuição dos dados de quando o modelo foi treinado. Se houver uma divergência alta (fenômeno conhecido como **Data Drift**), a equipe de dados recebe um alerta indicando que o modelo pode estar perdendo performance e precisa ser retreinado com informações mais recentes.
