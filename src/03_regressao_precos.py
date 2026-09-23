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
                             accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)
import warnings
from scipy.stats import randint
import os
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# Configuração de caminhos robustos
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "porto_imoveis_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ANÁLISE PREDITIVA PREÇOS IMÓVEIS PORTO - PIPELINE DE MACHINE LEARNING")
print("="*80 + "\n")

# ============================================================================
# 1. CARREGAR E VALIDAR DADOS
# ============================================================================
print("1. Carregando dataset limpo...")
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"❌ Ficheiro '{DATASET_PATH}' não encontrado.\n"
        f"Execute primeiro '02_preparar_dataset.py'."
    )

df = pd.read_csv(DATASET_PATH)
print(f"Dataset: {len(df):,} imóveis Porto")

# VALIDAÇÃO DE SCHEMA (evita erros se colunas em falta)
required_num = ['size', 'rooms', 'bathrooms', 'numPhotos',
                'dist_centro_km', 'm2_por_quarto', 'densidade_wc']
required_cat = ['zona_geografica', 'status', 'floor', 'localizacao_premium']
missing_cols = [c for c in required_num + required_cat if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"❌ As seguintes colunas estão em falta no dataset limpo: {missing_cols}.\n"
        f"Certifica-te que executaste 02_preparar_dataset.py antes deste script."
    )

print(f"✅ Schema validado. Todas as colunas necessárias estão presentes.\n")

# Feature Engineering
df['priceByArea'] = df['price'] / df['size']
median_price = df['priceByArea'].median()
df['PriceCategory'] = (df['priceByArea'] > median_price).astype(int)
print(f"Mediana €/m²: €{median_price:.0f} (limiar barato/caro)")

# Features
num_features = required_num
cat_features = required_cat
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
    
    # R² treino para análise overfitting
    r2_train = r2_score(y_train_reg, model.predict(X_train))
    reg_results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'R²_train': r2_train}
    print(f"  R²_test: {r2:.3f} | R²_train: {r2_train:.3f} | RMSE: €{rmse:.0f} | MAE: €{mae:.0f}")

# ============================================================================
# 4. OTIMIZAÇÃO RANDOM FOREST COM REGULARIZAÇÃO ANTI-OVERFITTING
# ============================================================================
print("\n" + "="*50)
print("🔧 OTIMIZAÇÃO Random Forest (Grid Search Regularizado)")
print("="*50)

# TESTE 1: Modelo BASE
print("\n🔹 Teste 1: Random Forest BASE")
rf_base = reg_models['Random Forest']
rf_base.fit(X_train, y_train_reg)
r2_base_test = r2_score(y_test_reg, rf_base.predict(X_test))
r2_base_cv = cross_val_score(rf_base, X, y_reg, cv=5, scoring='r2').mean()
print(f"  R² test: {r2_base_test:.3f} | R² CV: {r2_base_cv:.3f} | GAP: {r2_base_test - r2_base_cv:.3f}")

# TESTE 2: Modelo REGULARIZADO
print("\n🔹 Teste 2: Random Forest REGULARIZADO")
param_dist_reg = {
    'model__n_estimators': randint(100, 200),
    'model__max_depth': randint(8, 14),
    'model__min_samples_split': randint(8, 20),
    'model__min_samples_leaf': randint(4, 10)
}

rf_opt = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist_reg,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_opt.fit(X_train, y_train_reg)
print(f"\n✅ Melhor R² (CV 5-fold): {rf_opt.best_score_:.3f}")
print(f"✅ Melhores params: {rf_opt.best_params_}")

# RECALCULAR métricas do RF REGULARIZADO
rf_regularized = rf_opt.best_estimator_
y_pred_rf_reg = rf_regularized.predict(X_test)
r2_rf_test = r2_score(y_test_reg, y_pred_rf_reg)
rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_reg))
mae_rf = mean_absolute_error(y_test_reg, y_pred_rf_reg)
r2_rf_train = r2_score(y_train_reg, rf_regularized.predict(X_train))
r2_rf_cv = cross_val_score(rf_regularized, X, y_reg, cv=5, scoring='r2').mean()

print(f"\n📊 COMPARAÇÃO MODELOS:")
print(f"  BASE:         R² test={r2_base_test:.3f} | R² CV={r2_base_cv:.3f} | GAP={r2_base_test - r2_base_cv:.3f}")
print(f"  REGULARIZADO: R² test={r2_rf_test:.3f} | R² CV={r2_rf_cv:.3f} | GAP={r2_rf_test - r2_rf_cv:.3f}")
print(f"\n🏆 GANHO GENERALIZAÇÃO: {r2_rf_cv - r2_base_cv:+.3f} (R² CV)")

# ATUALIZAR dicionário com modelo REGULARIZADO
reg_models['Random Forest'] = rf_regularized
reg_results['Random Forest'] = {
    'R²': r2_rf_test, 
    'RMSE': rmse_rf, 
    'MAE': mae_rf, 
    'R²_train': r2_rf_train,
    'R²_CV': r2_rf_cv
}

print(f"\nRandom Forest REGULARIZADO | R² test: {r2_rf_test:.3f} | R² CV: {r2_rf_cv:.3f} | RMSE: €{rmse_rf:.0f} | MAE: €{mae_rf:.0f}")

# ============================================================================
# 5. OPORTUNIDADES INVESTIMENTO (apenas conjunto de TESTE + filtro incerteza)
# ============================================================================
print("\n" + "="*50)
print("💰 TOP OPORTUNIDADES (apenas conjunto de TESTE)")
print("="*50)

best_reg = reg_models['Random Forest']

# resíduos no teste
y_pred_test = best_reg.predict(X_test)
residuos_test = y_test_reg.values - y_pred_test
resid_std = residuos_test.std()
print(f"Desvio-padrão dos resíduos (teste): {resid_std:.2f} €/m²")

# construir DataFrame alinhado com X_test
df_test = X_test.copy()
df_test['priceByArea'] = y_test_reg.values
df_test['Predicted'] = y_pred_test
df_test['Diferenca'] = df_test['priceByArea'] - df_test['Predicted']
df_test['Economia'] = np.maximum(0, -df_test['Diferenca'])

# filtrar apenas casos realmente fora do ruído (resíduo < -1*std)
df_oportunidades = df_test[df_test['Diferenca'] < -resid_std].copy()

if len(df_oportunidades) == 0:
    print("⚠️ Nenhuma oportunidade clara (acima de 1 desvio-padrão) no conjunto de teste.")
else:
    top_oportunidades = df_oportunidades.nlargest(10, 'Economia')
    print(f"\n🎯 Encontradas {len(df_oportunidades)} oportunidades potenciais (resíduo < -1 std).")
    print(f"📋 TOP 10 OPORTUNIDADES:\n")
    for i, (_, row) in enumerate(top_oportunidades.iterrows(), 1):
        print(
            f"{i:2d}. 💎 Economia: €{row['Economia']:.0f}/m² | "
            f"Size: {row['size']:.0f}m² | Zona {row.get('zona_geografica', 'N/A')} | "
            f"{row['dist_centro_km']:.2f}km centro"
        )
    
    # salvar oportunidades em CSV
    output_oportunidades = RESULTS_DIR / 'oportunidades_teste.csv'
    top_oportunidades.to_csv(output_oportunidades, index=False)
    print(f"\n✅ Salvo: {output_oportunidades}")

# ============================================================================
# 6. CLASSIFICAÇÃO (Barato vs Caro)
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
# 7. THRESHOLD ANALYSIS (gerar CSV automaticamente)
# ============================================================================
print("\n" + "="*50)
print("📈 ANÁLISE DE THRESHOLDS (Random Forest Classificação)")
print("="*50)

y_proba = rf_class.predict_proba(X_test)[:, 1]
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
rows = []

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    acc_thr = accuracy_score(y_test_class, y_pred_thr)
    prec_thr = precision_score(y_test_class, y_pred_thr, zero_division=0)
    rec_thr = recall_score(y_test_class, y_pred_thr, zero_division=0)
    f1_thr = f1_score(y_test_class, y_pred_thr, zero_division=0)
    
    # calcular FPR manualmente
    cm_thr = confusion_matrix(y_test_class, y_pred_thr)
    tn, fp, fn, tp = cm_thr.ravel()
    fpr_thr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    rows.append({
        'Threshold': thr,
        'TPR_Recall': rec_thr,
        'FPR': fpr_thr,
        'Precision': prec_thr,
        'Accuracy': acc_thr,
        'F1-Score': f1_thr
    })
    print(f"Thr={thr:.1f} | Recall={rec_thr:.3f} | FPR={fpr_thr:.3f} | Prec={prec_thr:.3f} | F1={f1_thr:.3f}")

thr_df = pd.DataFrame(rows)
output_thresholds = RESULTS_DIR / 'threshold_analysis_rf_class.csv'
thr_df.to_csv(output_thresholds, index=False)
print(f"\n✅ Salvo: {output_thresholds}")

# ============================================================================
# 8. VISUALIZAÇÕES COMPLETAS
# ============================================================================
print("\n📈 Gerando gráficos...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Completa Preços Imóveis Porto', fontsize=16, fontweight='bold')

# Gráfico 1: R² Regressão
reg_df = pd.DataFrame(reg_results).T[['R²', 'RMSE', 'MAE']].sort_values('R²', ascending=False)
axes[0,0].barh(reg_df.index, reg_df['R²'], color='skyblue')
axes[0,0].set_title('🏆 R² Regressão (Teste)')
axes[0,0].set_xlabel('Score')
for i, v in enumerate(reg_df['R²']):
    axes[0,0].text(v+0.01, i, f'{v:.3f}', va='center')

# Gráfico 2: Acurácia Classificação
class_df = pd.DataFrame(class_results).T[['Accuracy', 'F1-Score']].sort_values('Accuracy', ascending=False)
axes[0,1].barh(class_df.index, class_df['Accuracy'], color='lightgreen')
axes[0,1].set_title('📊 Accuracy Classificação')
axes[0,1].set_xlabel('Score')
for i, v in enumerate(class_df['Accuracy']):
    axes[0,1].text(v+0.01, i, f'{v:.3f}', va='center')

# Gráfico 3: Real vs Previsto (RF REGULARIZADO)
axes[1,0].scatter(y_test_reg, y_pred_rf_reg, alpha=0.6)
axes[1,0].plot([y_test_reg.min(), y_test_reg.max()],
               [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1,0].set_xlabel('Real (€/m²)'), axes[1,0].set_ylabel('Previsto (€/m²)')
axes[1,0].set_title(f'RF REGULARIZADO: Real vs Previsto (R²={r2_rf_test:.3f})')

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
output_painel = RESULTS_DIR / 'resultados_completos.png'
plt.savefig(output_painel, dpi=300, bbox_inches='tight')
print(f"✅ Salvo: {output_painel}")

# ============================================================================
# 9. SALVAR RESULTADOS CSV
# ============================================================================
# Adicionar R² CV aos resultados de regressão
for model_name in ['Linear Regression', 'LinearSVR']:
    if model_name in reg_models:
        cv_scores = cross_val_score(reg_models[model_name], X, y_reg, cv=5, scoring='r2')
        reg_results[model_name]['R²_CV'] = cv_scores.mean()

output_comparison = RESULTS_DIR / 'comparison.csv'
output_classification = RESULTS_DIR / 'classification.csv'
pd.DataFrame(reg_results).T.to_csv(output_comparison)
pd.DataFrame(class_results).T.drop(columns=['Confusion Matrix']).to_csv(output_classification)
print(f"✅ Salvo: {output_comparison} | {output_classification} (com R²_CV)\n")

print("="*80)
print("🎉 ANÁLISE COMPLETA! Execute 04_relatorio_final.py para gerar relatório.")
print("="*80)
