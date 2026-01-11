# Previsão Preços Imóveis Porto - Idealista.pt

**Engenharia Informática | ISLA 2025/2026**  
**Projeto Final - Inteligência Artificial**

## 📋 Descrição

Análise preditiva de preços de imóveis no Porto utilizando dados reais da API Idealista.pt. Implementação de seis algoritmos de Machine Learning (três regressão + três classificação) com validação cruzada, otimização de hiperparâmetros e identificação de oportunidades de investimento.

## 🎯 Objetivos

- **Regressão**: Prever preço por metro quadrado (€/m²) de imóveis
- **Classificação**: Classificar imóveis como "barato" ou "caro" (abaixo/acima mediana)
- **Análise**: Identificar top 10 oportunidades de investimento (imóveis subvalorizados)

## 📊 Dataset

- **Fonte**: Idealista.pt API v3.5 (OAuth2)
- **Região**: Porto e arredores (raio 7km)
- **Tamanho**: 4324 imóveis após limpeza
- **Período**: Janeiro 2026
- **Features**: size, rooms, bathrooms, numPhotos, neighborhood, status, floor

## 🛠️ Tecnologias

- **Python 3.12**
- **Bibliotecas**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Algoritmos**:
  - Regressão: Linear Regression, Random Forest Regressor, LinearSVR
  - Classificação: Logistic Regression, Random Forest Classifier, LinearSVC

## 📁 Estrutura do Projeto

housing-market-ml-idealista/
│
├── datasets/
│ └── porto_imoveis_dataset.csv # Dataset final (4324 imóveis)
│
├── src/
│ ├── 01_criar_dataset.py # Coleta dados API (100 requests/mês)
│ ├── 02_preparar_dataset.py # Limpeza e feature engineering
│ ├── 03_regressao_precos.py # Modelos ML completos
│ └── 04_relatorio_final.py # Geração relatório e gráficos
│
├── results/
│ ├── comparison.csv # Métricas regressão
│ ├── classification.csv # Métricas classificação
│ ├── resultados_completos.png # Painel 4 gráficos
│ ├── RELATORIO_GRAFICO_FINAL.png # Comparação modelos
│ ├── MATRIZ_CONFUSAO.png # Matriz confusão RF
│ └── RELATORIO_FINAL_PRECOS_PORTO.txt # Relatório completo
│
└── README.md


## 🚀 Como Executar

### 1. Instalar Dependências

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests scipy
```

### 2. Executar Pipeline Completo
# Opção A: Coletar novos dados (requer API key válida)
cd src
Dar run a 01_criar_dataset.py     # ~5000 imóveis (100 requests)
Dar run a 02_preparar_dataset.py  # Limpeza → 4324 imóveis
Dar run a 03_regressao_precos.py  # Treino modelos (5-10 min)
Dar run a 04_relatorio_final.py   # Gerar relatório

# Opção B: Usar dataset existente (pular passo 1)
Dar run a 02_preparar_dataset.py
Dar run a 03_regressao_precos.py
Dar run a 04_relatorio_final.py

### 3. Verificar Resultados

Ficheiros gerados em results/:

    RELATORIO_FINAL_PRECOS_PORTO.txt → Copiar para Word

    RELATORIO_GRAFICO_FINAL.png → Inserir no relatório

    MATRIZ_CONFUSAO.png → Inserir no relatório

    resultados_completos.png → Análise completa 4 painéis

🔮 Trabalho Futuro

    Implementar XGBoost/LightGBM (alvo R²>0.50)

    Features geoespaciais (distância metro/centro)

    Análise temporal preços (séries temporais)

    App web Streamlit para previsões real-time

    Integração Google Maps API (visualização)

📚 Referências

    Idealista.pt API v3.5 - Property Search Documentation

    Scikit-learn 1.6.1 - Machine Learning Library

    Documentação algoritmos (REGRESSAO_LINEAR.pdf, Random_Forest.pdf, SVM.pdf)

👥 Autores

Dário Soares - a22307370
Engenharia Informática - ISLA
Inteligência Artificial 2025/2026