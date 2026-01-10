# -*- coding: utf-8 -*-
"""
03_regressao_precos.py - ANÁLISE PREDITIVA COMPLETA
Regressão (Linear, RF, SVR) + Classificação (LogReg, RF, SVC)
Validação cruzada, otimização hiperparâmetros, oportunidades investimento
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, confusion_matrix, classification_report)
import warnings
from scipy.stats import randint
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

if not os.path.exists('../results'):
    os.makedirs('../results')
    print("✅ Diretório ../results/ criado\n")
    
print("="*80)
print("ANÁLISE PREDITIVA PREÇOS IMÓVEIS PORTO - VERSÃO FINAL")
print("="*80 + "\n")

# ============================================================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================================================
print("1. Carregando dataset limpo...")
df = pd.read_csv('../datasets/porto_imoveis_dataset.csv')
print(f"Dataset: {len(df):,} imóveis Porto")

# Feature Engineering
df['priceByArea'] = df['price'] / df['size']
median_price = df['priceByArea'].median()
df['PriceCategory'] = (df['priceByArea'] > median_price).astype(int)
print(f"Mediana €/m²: €{median_price:.0f} (limiar barato/caro)")

# Features
#num_features = ['size', 'rooms', 'bathrooms', 'numPhotos']
#cat_features = ['neighborhood', 'status', 'floor']
num_features = ['size', 'rooms', 'bathrooms', 'numPhotos', 
                'dist_centro_km', 'm2_por_quarto', 'densidade_wc']
cat_features = ['zona_geografica', 'status', 'floor', 'localizacao_premium']
X = df[num_features + cat_features]
y_reg = df['priceByArea']
y_class = df['PriceCategory']

# ============================================================================
# 2. PIPELINES PRÉ-PROCESSAMENTO
# ============================================================================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=42, stratify=y_class
)
print(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}\n")

# ============================================================================
# 3. REGRESSÃO (Linear + RF + SVR)
# ============================================================================
print("="*50)
print("REGRESSÃO - Previsão €/m²")
print("="*50)

reg_models = {
    'Linear Regression': Pipeline([('prep', preprocessor), ('model', LinearRegression())]),
    'Random Forest': Pipeline([('prep', preprocessor), 
                               ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]),
    'LinearSVR': Pipeline([('prep', preprocessor), ('model', LinearSVR(random_state=42, max_iter=1000))])
}

reg_results = {}
for name, model in reg_models.items():
    print(f"Treinando {name}...")
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    mae = mean_absolute_error(y_test_reg, y_pred)
    
    reg_results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}
    print(f"  R²: {r2:.3f} | RMSE: €{rmse:.0f} | MAE: €{mae:.0f}")

# Validação cruzada RF
print("\n🔍 Validação Cruzada 5-fold (Random Forest):")
rf_cv = cross_val_score(reg_models['Random Forest'], X, y_reg, cv=5, scoring='r2')
print(f"  R² médio: {rf_cv.mean():.3f} (±{rf_cv.std():.3f})")

# ============================================================================
# 4. CLASSIFICAÇÃO (Barato vs Caro)
# ============================================================================
print("\n" + "="*50)
print("CLASSIFICAÇÃO - Barato (0) vs Caro (1)")
print("="*50)

class_models = {
    'Logistic Regression': Pipeline([('prep', preprocessor), 
                                      ('model', LogisticRegression(random_state=42, max_iter=1000))]),
    'Random Forest': Pipeline([('prep', preprocessor), 
                               ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]),
    'LinearSVC': Pipeline([('prep', preprocessor), ('model', LinearSVC(random_state=42, max_iter=1000))])
}

class_results = {}
for name, model in class_models.items():
    print(f"Treinando {name}...")
    model.fit(X_train, y_train_class)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test_class, y_pred)
    report = classification_report(y_test_class, y_pred, output_dict=True)
    
    class_results[name] = {
        'Accuracy': acc,
        'F1-Score': report['1']['f1-score'],
        'Confusion Matrix': confusion_matrix(y_test_class, y_pred)
    }
    print(f"  Acurácia: {acc:.3f} | F1: {report['1']['f1-score']:.3f}")

rf_class = class_models['Random Forest']
cm = confusion_matrix(y_test_class, rf_class.predict(X_test))
print(f"\n📊 Matriz Confusão RF (Accuracy {class_results['Random Forest']['Accuracy']:.1%}):")
print(f"  [[{cm[0,0]:3d} {cm[0,1]:3d}]  ← Barato")
print(f"   [{cm[1,0]:3d} {cm[1,1]:3d}]]  ← Caro")

# ============================================================================
# 5. OTIMIZAÇÃO RANDOM FOREST
# ============================================================================
print("\n" + "="*50)
print("🔧 OTIMIZAÇÃO Random Forest (RandomizedSearchCV)")
print("="*50)

param_dist = {
    'model__n_estimators': randint(80, 250),
    'model__max_depth': randint(6, 22),
    'model__min_samples_split': randint(2, 12)
}

rf_base = reg_models['Random Forest']  # pipeline RF base

rf_opt = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

rf_opt.fit(X_train, y_train_reg)
print(f"Melhor R² (CV): {rf_opt.best_score_:.3f}")
print(f"Melhores params: {rf_opt.best_params_}")

# substituir RF no dicionário pelos melhores parâmetros
reg_models['Random Forest'] = rf_opt.best_estimator_

# recalcular métricas do RF otimizado no conjunto de teste
print("\n🔁 Recalculando métricas com Random Forest OTIMIZADO...")
y_pred_rf_opt = reg_models['Random Forest'].predict(X_test)
r2_rf = r2_score(y_test_reg, y_pred_rf_opt)
rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_opt))
mae_rf = mean_absolute_error(y_test_reg, y_pred_rf_opt)

reg_results['Random Forest'] = {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf}
print(f"Random Forest (OTIMIZADO) | R²: {r2_rf:.3f} | RMSE: €{rmse_rf:.0f} | MAE: €{mae_rf:.0f}")

# ============================================================================
# 6. OPORTUNIDADES INVESTIMENTO (subvalorizados, apenas TESTE)
# ============================================================================
print("\n" + "="*50)
print("💰 TOP OPORTUNIDADES (apenas conjunto de TESTE)")
print("="*50)

best_reg = reg_models['Random Forest']

# previsões em CV para estimar dispersão típica dos resíduos
cv_scores = cross_val_score(best_reg, X, y_reg, cv=5, scoring='r2')
print(f"R² CV (RF otimizado): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# resíduos no teste
y_pred_test = best_reg.predict(X_test)
residuos_test = y_test_reg - y_pred_test
resid_std = residuos_test.std()
print(f"Desvio-padrão dos resíduos (teste): {resid_std:.2f} €/m²")

# construir DataFrame alinhado com X_test
df_test = X_test.copy()
df_test['priceByArea'] = y_test_reg
df_test['Predicted'] = y_pred_test
df_test['Diferenca'] = df_test['priceByArea'] - df_test['Predicted']
df_test['Economia'] = np.maximum(0, -df_test['Diferenca'])

# filtrar apenas casos realmente fora do ruído (resíduo < -1*std)
df_oportunidades = df_test[df_test['Diferenca'] < -resid_std].copy()

if len(df_oportunidades) == 0:
    print("⚠️ Nenhuma oportunidade clara (acima de 1 desvio-padrão) no conjunto de teste.")
else:
    top_oportunidades = df_oportunidades.nlargest(10, 'Economia')
    for i, (_, row) in enumerate(top_oportunidades.iterrows(), 1):
        print(
            f"{i:2d}. 💎 Economia: €{row['Economia']:.0f}/m² | "
            f"Size: {row['size']:.0f}m² | Zona {row['zona_geografica']} | "
            f"{row['dist_centro_km']:.2f}km centro"
        )

# opcional: guardar oportunidades em CSV
top_oportunidades.to_csv('../results/oportunidades_teste.csv', index=False)
print("✅ Salvo: ../results/oportunidades_teste.csv")


# ============================================================================
# 7. VISUALIZAÇÕES COMPLETAS
# ============================================================================
print("\n📈 Gerando gráficos...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Completa Preços Imóveis Porto', fontsize=16, fontweight='bold')

# Gráfico 1: R² Regressão
reg_df = pd.DataFrame(reg_results).T.sort_values('R²', ascending=False)
axes[0,0].barh(reg_df.index, reg_df['R²'], color='skyblue')
axes[0,0].set_title('🏆 R² Regressão')
axes[0,0].set_xlabel('Score')
for i, v in enumerate(reg_df['R²']):
    axes[0,0].text(v+0.01, i, f'{v:.3f}', va='center')

# Gráfico 2: Acurácia Classificação
class_df = pd.DataFrame(class_results).T.sort_values('Accuracy', ascending=False)
axes[0,1].barh(class_df.index, class_df['Accuracy'], color='lightgreen')
axes[0,1].set_title('📊 Accuracy Classificação')
axes[0,1].set_xlabel('Score')
for i, v in enumerate(class_df['Accuracy']):
    axes[0,1].text(v+0.01, i, f'{v:.3f}', va='center')

# Gráfico 3: Real vs Previsto (RF)
axes[1,0].scatter(y_test_reg, reg_models['Random Forest'].predict(X_test), alpha=0.6)
axes[1,0].plot([y_test_reg.min(), y_test_reg.max()], 
               [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1,0].set_xlabel('Real'), axes[1,0].set_ylabel('Previsto')
axes[1,0].set_title('RF: Real vs Previsto')

# Gráfico 4: Matriz Confusão
im = axes[1,1].imshow(cm, cmap='Blues')
axes[1,1].set_title('Matriz Confusão RF Classificação')
for i in range(2):
    for j in range(2):
        axes[1,1].text(j, i, cm[i,j], ha='center', va='center', 
                      fontsize=20, color='white' if cm[i,j] > cm.max()/2 else 'black')
axes[1,1].set_xticks([0,1]), axes[1,1].set_yticks([0,1])
axes[1,1].set_xticklabels(['Barato','Caro'])
axes[1,1].set_yticklabels(['Barato','Caro'])
axes[1,1].set_xlabel('Previsto'), axes[1,1].set_ylabel('Real')

plt.tight_layout()
plt.savefig('../results/resultados_completos.png', dpi=300, bbox_inches='tight')
print("✅ Salvo: ../results/resultados_completos.png")

# ============================================================================
# 8. SALVAR RESULTADOS CSV
# ============================================================================
pd.DataFrame(reg_results).T.to_csv('../results/comparison.csv')
pd.DataFrame(class_results).T.to_csv('../results/classification.csv')
print("✅ Salvo: ../results/comparison.csv | classification.csv\n")

print("="*80)
print("🎉 ANÁLISE COMPLETA! Execute 04_relatorio_final.py para gerar relatório.")
print("="*80)
