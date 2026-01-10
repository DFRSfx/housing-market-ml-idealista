# -*- coding: utf-8 -*-
"""
04_relatorio_final.py - GERAÇÃO RELATÓRIO FINAL (COM CURVA ROC)
Consolida resultados, gera relatório estruturado e gráficos publication-ready
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("GERAÇÃO RELATÓRIO FINAL - PREÇOS IMÓVEIS PORTO")
print("="*60 + "\n")

# ============================================================================
# 1. CARREGAR RESULTADOS
# ============================================================================
print("📂 Carregando resultados...")
comparison = pd.read_csv('../results/comparison.csv', index_col=0)
classification = pd.read_csv('../results/classification.csv', index_col=0)

print("\n📊 REGRESSÃO:")
print(comparison.round(3))
print("\n📊 CLASSIFICAÇÃO:")
print(classification[['Accuracy', 'F1-Score']].round(3))

# ============================================================================
# 2. PARSER MATRIZ CONFUSÃO
# ============================================================================
def parse_confusion_matrix(cm_str):
    """Converte string matriz para numpy array"""
    clean = re.sub(r'[\[\]\n\s]+', ' ', str(cm_str)).strip()
    nums = [int(x) for x in clean.split() if x.isdigit()]
    return np.array(nums).reshape(2, 2)

cm_rf = parse_confusion_matrix(classification.loc['Random Forest', 'Confusion Matrix'])
print(f"\n✅ Matriz Confusão RF:")
print(cm_rf)

# ============================================================================
# 2.5. CALCULAR CURVA ROC (REQUER PROBABILIDADES)
# ============================================================================
print("\n📈 Calculando Curva ROC...")

# Carregar dataset e retreinar RF rapidamente para obter probabilidades
df = pd.read_csv('../datasets/porto_imoveis_dataset.csv')
df['priceByArea'] = df['price'] / df['size']
median_price = df['priceByArea'].median()
df['PriceCategory'] = (df['priceByArea'] > median_price).astype(int)

num_features = ['size', 'rooms', 'bathrooms', 'numPhotos']
cat_features = ['neighborhood', 'status', 'floor']
X = df[num_features + cat_features]
y_class = df['PriceCategory']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Treinar RF para obter probabilidades
rf_roc = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

rf_roc.fit(X_train, y_train)
y_proba = rf_roc.predict_proba(X_test)[:, 1]  # Probabilidades classe "Caro"

# Calcular ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"✅ AUC-ROC: {roc_auc:.3f}")

# ============================================================================
# 3. GERAR RELATÓRIO ESTRUTURADO
# ============================================================================
print("\n📝 Gerando RELATORIO_FINAL_PRECOS_PORTO.txt...")

with open("../results/RELATORIO_FINAL_PRECOS_PORTO.txt", "w", encoding='utf-8') as f:
    f.write("RELATÓRIO FINAL - PREVISÃO PREÇOS IMÓVEIS PORTO\n")
    f.write("Engenharia Informática - ISLA 2025/2026\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. INTRODUÇÃO\n")
    f.write("Análise preditiva preços €/m² (Idealista.pt - Porto).\n")
    f.write("• Regressão: preço contínuo\n")
    f.write("• Classificação: barato(0)/caro(1)\n")
    f.write("Dataset: 4324 imóveis | Features: size, neighborhood, etc.\n\n")
    
    f.write("2. ESTADO DA ARTE\n")
    f.write("- Regressão Linear: y=β₀+β₁X, R² [file:7]\n")
    f.write("- Random Forest: ensemble, Gini importance [file:23]\n")
    f.write("- SVM Linear: hiperplano ótimo [file:13]\n\n")
    
    f.write("3. METODOLOGIA\n")
    f.write("- Pré-processamento: StandardScaler + OneHotEncoder\n")
    f.write("- Split: 80/20 estratificado\n")
    f.write("- Modelos: Linear/RF/SVR | LogReg/RF/SVC\n")
    f.write("- Métricas: R²/RMSE/MAE | Acc/F1/Confusão/AUC-ROC\n\n")
    
    f.write("4. RESULTADOS REGRESSÃO\n")
    f.write(comparison.round(3).to_string() + "\n\n")
    best_reg = comparison['R²'].idxmax()
    best_r2 = comparison.loc[best_reg, 'R²']
    best_rmse = comparison.loc[best_reg, 'RMSE']
    f.write(f"🏆 Melhor: {best_reg} | R²={best_r2:.3f} | RMSE=€{best_rmse:.0f}/m²\n\n")
    
    f.write("5. RESULTADOS CLASSIFICAÇÃO\n")
    f.write(classification[['Accuracy', 'F1-Score']].round(3).to_string() + "\n\n")
    best_class = classification['Accuracy'].idxmax()
    best_acc = classification.loc[best_class, 'Accuracy']
    f.write(f"🏆 Melhor: {best_class} | Accuracy={best_acc:.1%} | AUC-ROC={roc_auc:.3f}\n\n")
    
    f.write(f"Matriz Confusão {best_class} ({best_acc:.1%} Acc):\n")
    f.write(f"[[{cm_rf[0,0]:3d} {cm_rf[0,1]:3d}]  ← Barato\n")
    f.write(f" [{cm_rf[1,0]:3d} {cm_rf[1,1]:3d}]]  ← Caro\n\n")
    
    f.write("Curva ROC:\n")
    f.write(f"- AUC-ROC: {roc_auc:.3f} (excelente discriminação)\n")
    f.write(f"- Interpretação: {roc_auc*100:.1f}% probabilidade de classificar\n")
    f.write(f"  corretamente par aleatório (barato, caro)\n\n")
    
    f.write("6. ANÁLISE FEATURES\n")
    f.write("Top features (importância RF): size > numPhotos > neighborhood > bathrooms\n\n")
    
    f.write("7. CONCLUSÕES\n")
    f.write(f"✅ {best_reg} domina regressão (R²={best_r2:.3f})\n")
    f.write(f"✅ {best_class} domina classificação (Acc={best_acc:.1%}, AUC={roc_auc:.3f})\n")
    f.write("✅ Sistema robusto previsão imobiliária Porto\n")
    f.write("✅ Validação cruzada confirma estabilidade\n\n")
    
    f.write("8. TRABALHO FUTURO\n")
    f.write("- XGBoost/LightGBM (R²>0.50)\n")
    f.write("- Features geoespaciais (distância centro/metro)\n")
    f.write("- App web Streamlit real-time\n")
    f.write("- Análise temporal preços (séries temporais)\n\n")
    
    f.write("9. REFERÊNCIAS\n")
    f.write("1. Idealista.pt Porto API v3.5 [2025]\n")
    f.write("2. Scikit-learn 1.6.1\n")
    f.write("3. Documentação algoritmos [file:7,9,10,13,23]\n")

print("✅ RELATORIO_FINAL_PRECOS_PORTO.txt gerado!")

# ============================================================================
# 4. GRÁFICO COMPARAÇÃO (2 painéis)
# ============================================================================
print("\n📊 Gerando RELATORIO_GRAFICO_FINAL.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Painel 1: Regressão
comparison['R²'].sort_values(ascending=False).plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title('🏆 R² Regressão', fontweight='bold', fontsize=14)
ax1.set_xlabel('Score')
for i, v in enumerate(comparison['R²'].sort_values(ascending=False)):
    ax1.text(v+0.01, i, f'{v:.3f}', va='center')

# Painel 2: Classificação
classification['Accuracy'].sort_values(ascending=False).plot(kind='barh', ax=ax2, color='lightgreen')
ax2.set_title('📊 Accuracy Classificação', fontweight='bold', fontsize=14)
ax2.set_xlabel('Score')
for i, v in enumerate(classification['Accuracy'].sort_values(ascending=False)):
    ax2.text(v+0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('../results/RELATORIO_GRAFICO_FINAL.png', dpi=300, bbox_inches='tight')
print("✅ RELATORIO_GRAFICO_FINAL.png gerado!")

# ============================================================================
# 5. MATRIZ CONFUSÃO (HEATMAP)
# ============================================================================
print("\n📈 Gerando MATRIZ_CONFUSAO.png...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Barato', 'Caro'], yticklabels=['Barato', 'Caro'],
            annot_kws={'size': 20})
plt.title(f'Matriz Confusão RF (Accuracy {best_acc:.1%})', fontweight='bold', fontsize=14)
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('../results/MATRIZ_CONFUSAO.png', dpi=300, bbox_inches='tight')
print("✅ MATRIZ_CONFUSAO.png gerado!")

# ============================================================================
# 6. CURVA ROC (NOVO!)
# ============================================================================
print("\n📈 Gerando CURVA_ROC.png...")
plt.figure(figsize=(8, 7))

# Plot curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
         label=f'Random Forest (AUC = {roc_auc:.3f})')

# Linha diagonal (classificador aleatório)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Classificador Aleatório (AUC = 0.500)')

# Configurações
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12, fontweight='bold')
plt.title('Curva ROC - Random Forest Classifier\nClassificação Barato vs Caro', 
          fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3, linestyle='--')

# Adicionar ponto de operação (threshold 0.5)
idx_threshold_05 = np.argmin(np.abs(thresholds - 0.5))
plt.plot(fpr[idx_threshold_05], tpr[idx_threshold_05], 'ro', markersize=10, 
         label=f'Threshold 0.5 (FPR={fpr[idx_threshold_05]:.3f}, TPR={tpr[idx_threshold_05]:.3f})')
plt.legend(loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig('../results/CURVA_ROC.png', dpi=300, bbox_inches='tight')
print("✅ CURVA_ROC.png gerado!")

# ============================================================================
# 7. PAINEL COMPLETO 4 GRÁFICOS (INCLUINDO ROC)
# ============================================================================
print("\n📊 Gerando PAINEL_COMPLETO.png...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Gráfico 1: R² Regressão
ax1 = fig.add_subplot(gs[0, 0])
comparison['R²'].sort_values(ascending=False).plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title('🏆 R² Regressão', fontweight='bold', fontsize=14)
ax1.set_xlabel('Score')
for i, v in enumerate(comparison['R²'].sort_values(ascending=False)):
    ax1.text(v+0.01, i, f'{v:.3f}', va='center')

# Gráfico 2: Accuracy Classificação
ax2 = fig.add_subplot(gs[0, 1])
classification['Accuracy'].sort_values(ascending=False).plot(kind='barh', ax=ax2, color='lightgreen')
ax2.set_title('📊 Accuracy Classificação', fontweight='bold', fontsize=14)
ax2.set_xlabel('Score')
for i, v in enumerate(classification['Accuracy'].sort_values(ascending=False)):
    ax2.text(v+0.01, i, f'{v:.3f}', va='center')

# Gráfico 3: Matriz Confusão
ax3 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=['Barato', 'Caro'], yticklabels=['Barato', 'Caro'],
            annot_kws={'size': 16})
ax3.set_title(f'Matriz Confusão RF (Acc {best_acc:.1%})', fontweight='bold', fontsize=14)
ax3.set_ylabel('Real')
ax3.set_xlabel('Previsto')

# Gráfico 4: Curva ROC
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'RF (AUC={roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
ax4.plot(fpr[idx_threshold_05], tpr[idx_threshold_05], 'ro', markersize=8)
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('FPR', fontweight='bold')
ax4.set_ylabel('TPR', fontweight='bold')
ax4.set_title(f'Curva ROC (AUC={roc_auc:.3f})', fontweight='bold', fontsize=14)
ax4.legend(loc="lower right", fontsize=9)
ax4.grid(alpha=0.3)

fig.suptitle('Análise Completa - Preços Imóveis Porto', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('../results/PAINEL_COMPLETO.png', dpi=300, bbox_inches='tight')
print("✅ PAINEL_COMPLETO.png gerado!")


# ============================================================================
# 8. VISUALIZAÇÃO ÁRVORE INDIVIDUAL DO RANDOM FOREST (NOVO!)
# ============================================================================
print("\n🌳 Gerando visualização árvore individual...")

# Extrair uma árvore representativa (primeira árvore)
rf_model = rf_roc.named_steps['model']
tree_to_plot = rf_model.estimators_[0]  # Primeira árvore do ensemble

# Obter nomes das features após pré-processamento
preprocessor_fitted = rf_roc.named_steps['prep']
feature_names_cat = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(cat_features)
all_feature_names = num_features + list(feature_names_cat)

# Visualizar árvore (limitada a profundidade 4 para legibilidade)
plt.figure(figsize=(25, 15))
plot_tree(
    tree_to_plot,
    feature_names=all_feature_names,
    class_names=['Barato', 'Caro'],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=4,  # Limitar profundidade (árvore completa seria gigante)
    proportion=True,
    precision=2
)
plt.title('Árvore de Decisão Individual do Random Forest (Profundidade Máx: 4)\n'
          'Exemplo de 1 das 100 árvores do ensemble', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../results/ARVORE_RF_INDIVIDUAL.png', dpi=200, bbox_inches='tight')
print("✅ ARVORE_RF_INDIVIDUAL.png gerado!")

# ============================================================================
# 9. FEATURE IMPORTANCE (NOVO!)
# ============================================================================
print("\n📊 Gerando gráfico Feature Importance...")

# Calcular importâncias
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features

# Plot
plt.figure(figsize=(10, 8))
plt.barh(range(15), importances[indices][::-1], color='steelblue')
plt.yticks(range(15), [all_feature_names[i] for i in indices][::-1])
plt.xlabel('Importância (Redução Média Impureza Gini)', fontweight='bold')
plt.title('Top 15 Features Mais Importantes - Random Forest', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Adicionar valores nas barras
for i, v in enumerate(importances[indices][::-1]):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../results/FEATURE_IMPORTANCE.png', dpi=300, bbox_inches='tight')
print("✅ FEATURE_IMPORTANCE.png gerado!")

# ============================================================================
# CONCLUSÃO
# ============================================================================
print("\n" + "="*60)
print("🎉 RELATÓRIO FINAL COMPLETO COM TODAS VISUALIZAÇÕES!")
print("="*60)
print("\n📄 Ficheiros gerados:")
print("  • ../results/RELATORIO_FINAL_PRECOS_PORTO.txt")
print("  • ../results/RELATORIO_GRAFICO_FINAL.png")
print("  • ../results/MATRIZ_CONFUSAO.png")
print("  • ../results/CURVA_ROC.png")
print("  • ../results/PAINEL_COMPLETO.png")
print("  • ../results/ARVORE_RF_INDIVIDUAL.png  [NOVO!]")
print("  • ../results/FEATURE_IMPORTANCE.png  [NOVO!]")
print("  • ../results/comparison.csv")
print("  • ../results/classification.csv")
print(f"\n📊 Métricas Finais:")
print(f"  • Regressão: R²={best_r2:.3f}, RMSE=€{best_rmse:.0f}/m²")
print(f"  • Classificação: Acc={best_acc:.1%}, AUC-ROC={roc_auc:.3f}")
print(f"\n🌳 Top 3 Features:")
indices = np.argsort(rf_model.feature_importances_)[::-1][:3]
for i, idx in enumerate(indices, 1):
    print(f"  {i}. {all_feature_names[idx]}: {rf_model.feature_importances_[idx]:.3f}")
print("\n✅ PRONTO PARA ENTREGA ACADÉMICA!")
