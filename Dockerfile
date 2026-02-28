# Usar imagem base leve de Python
FROM python:3.12-slim

# Variáveis ​​de ambiente
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Diretório de trabalho no container
WORKDIR /code

# Instalar dependências de sistema (se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar os requerimentos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os diretórios da aplicação, módulos e modelos
# NOTA: Em produção ideal, o modelo (.joblib) seria baixado de um bucket AWS S3/GCS.
COPY src/ /code/src/
COPY app/ /code/app/
COPY models/ /code/models/

# Expor a porta que a aplicação vai rodar
EXPOSE 8000

# Comando para rodar a aplicação usando Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
