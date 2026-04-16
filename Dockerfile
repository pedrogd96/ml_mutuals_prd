FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# criar pasta de logs
RUN mkdir -p logs

EXPOSE 5000

CMD ["python", "src/api/app.py"]