# 📊 Projeto de Machine Learning — Risco de Inadimplência em Contratos de Mútuo

## 🧠 Visão Geral
Este projeto tem como objetivo prever o risco de inadimplência em contratos de mútuo entre empresas utilizando machine learning com scikit-learn.

## 🎯 Problema de Negócio
Automatizar a análise de risco financeiro, reduzindo erros manuais e melhorando a tomada de decisão.

## 🤖 Objetivo do Modelo
Classificar contratos:
- 0 → Adimplente
- 1 → Inadimplente

## ⚙️ Tecnologias
- Python
- scikit-learn
- MLflow
- Flask
- Docker
- Grafana + Loki + Promtail

## 🚀 Subir o projeto
docker-compose up --build

## 🌐 URLs
API: http://localhost:5000
Grafana: http://localhost:3000 (admin/admin)
MLflow: http://localhost:5001

## 📡 Exemplo API
POST /predict

{
  "version": "v1",
  "data": {
    "valor_contrato": 2000000,
    "prazo_meses": 36,
    "taxa_juros": 0.09,
    "ebitda_margem": 0.12,
    "crescimento_receita_12m": 0.05,
    "volatilidade_resultados": 0.2,
    "liquidez": 1.4,
    "alavancagem": 0.7,
    "cobertura_juros": 3.5,
    "historico_atrasos_12m": 1,
    "variacao_capital_giro": 0.02,
    "estabilidade_fluxo_caixa": 0.15,
    "exposicao_cambial": 0.3,
    "sensibilidade_macro": 0.4,
    "rating_interno_score": 0.8,
    "garantia_ratio": 0.9
  }
}

Resposta:
{
  "prediction": 0
}

## 📊 Logs
logs/server-DD-MM-YYYY.log

## 🏁 Conclusão
Projeto completo de ML com foco em engenharia, versionamento e observabilidade.
