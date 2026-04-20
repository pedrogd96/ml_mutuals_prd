FROM python:3.11-slim

# Define diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas requirements primeiro (melhora cache)
COPY requirements.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Define PYTHONPATH para reconhecer src como pacote raiz
ENV PYTHONPATH=/app

# Cria diretórios necessários (evita erro em runtime)
RUN mkdir -p /app/logs /app/models/artifacts

# Porta da API
EXPOSE 5000

# Executa como módulo (ESSENCIAL)
CMD ["python", "-m", "src.api.app"]