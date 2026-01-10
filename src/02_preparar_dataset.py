# -*- coding: utf-8 -*-
"""
02_preparar_dataset.py - PRÉ-PROCESSAMENTO DATASET PORTO
Limpeza, remoção outliers, feature engineering
Input: porto_imoveis_dataset.csv (raw)
Output: porto_imoveis_dataset.csv (limpo)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

print("="*60)
print("PRÉ-PROCESSAMENTO DATASET IDEALISTA.PT PORTO")
print("="*60 + "\n")

# 1. Carregar dados raw
print("📂 Carregando porto_imoveis_dataset.csv...")
df = pd.read_csv("../datasets/porto_imoveis_dataset.csv")
print(f"Total imóveis raw: {len(df)}")

# 2. Remover duplicados (propertyCode)
df = df.drop_duplicates(subset=["propertyCode"], keep="first")
print(f"Após remover duplicados: {len(df)}")

# 3. Remover linhas sem preço/tamanho (essenciais)
df = df.dropna(subset=["price", "size"])
df = df[(df["price"] > 0) & (df["size"] > 10)]  # preço > 0, área > 10m²
print(f"Após validação preço/tamanho: {len(df)}")

# 4. Calcular priceByArea (€/m²)
if "priceByArea" not in df.columns or df["priceByArea"].isna().all():
    df["priceByArea"] = df["price"] / df["size"]
    print("✅ priceByArea calculado (price/size)")

# 5. Remover outliers extremos (percentis 1%-99%)
q1_price, q99_price = df["price"].quantile([0.01, 0.99])
df = df[(df["price"] >= q1_price) & (df["price"] <= q99_price)]

q1_pb, q99_pb = df["priceByArea"].quantile([0.01, 0.99])
df = df[(df["priceByArea"] >= q1_pb) & (df["priceByArea"] <= q99_pb)]
print(f"Após remover outliers (1%-99%): {len(df)}")

# ============================================================================
# 6. FEATURE ENGINEERING GEOGRÁFICO (NOVO!)
# ============================================================================
print("\n" + "="*60)
print("🌍 FEATURE ENGINEERING GEOGRÁFICO")
print("="*60)

# 6.1. Clustering geográfico (resolver problema dos 87% neighborhoods missing)
print("\n🔹 Criando zonas geográficas (KMeans clustering)...")
coords = df[["latitude", "longitude"]].values
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df["zona_geografica"] = kmeans.fit_predict(coords)
print(f"   ✅ {df['zona_geografica'].nunique()} zonas criadas")
print(f"   Distribuição: {df['zona_geografica'].value_counts().to_dict()}")

# 6.2. Distância ao centro do Porto (Avenida dos Aliados)
CENTRO_PORTO_LAT = 41.1496
CENTRO_PORTO_LON = -8.6109

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula distância em km entre dois pontos (fórmula Haversine)"""
    R = 6371  # Raio da Terra em km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

print("\n🔹 Calculando distância ao centro Porto...")
df["dist_centro_km"] = haversine_distance(
    df["latitude"], 
    df["longitude"],
    CENTRO_PORTO_LAT, 
    CENTRO_PORTO_LON
)
print(f"   ✅ Distância média: {df['dist_centro_km'].mean():.2f} km")
print(f"   Range: {df['dist_centro_km'].min():.2f} - {df['dist_centro_km'].max():.2f} km")

# 6.3. Features de interação
print("\n🔹 Criando features de interação...")
df["m2_por_quarto"] = df["size"] / (df["rooms"] + 1)  # +1 para evitar divisão por 0
df["preco_total_estimado"] = df["priceByArea"] * df["size"]  # Redundante mas útil para análise
df["densidade_wc"] = df["bathrooms"] / (df["rooms"] + 1)
print("   ✅ m2_por_quarto, preco_total_estimado, densidade_wc")

# 6.4. Flag localização premium (< 3km centro)
df["localizacao_premium"] = (df["dist_centro_km"] < 3.0).astype(int)
n_premium = df["localizacao_premium"].sum()
print(f"\n🔹 Imóveis premium (<3km centro): {n_premium} ({n_premium/len(df)*100:.1f}%)")

# ============================================================================
# 7. Estatísticas finais
# ============================================================================
print("\n" + "="*60)
print("ESTATÍSTICAS DATASET FINAL")
print("="*60)
print(f"Nº imóveis:        {len(df):,}")
print(f"Preço médio:       €{df['price'].mean():,.0f}")
print(f"Preço/m² médio:    €{df['priceByArea'].mean():,.0f}")
print(f"Preço/m² mediana:  €{df['priceByArea'].median():,.0f}")
print(f"Tamanho médio:     {df['size'].mean():.0f} m²")
print(f"Quartos médio:     {df['rooms'].mean():.1f}")
print(f"Dist. centro méd:  {df['dist_centro_km'].mean():.2f} km")

print("\nTop 5 zonas geográficas (por nº imóveis):")
print(df["zona_geografica"].value_counts().head())

print("\nStatus imóveis:")
print(df["status"].value_counts())

# 8. Salvar dataset limpo (sobrescreve)
df.to_csv("../datasets/porto_imoveis_dataset.csv", index=False)
print(f"\n✅ Dataset limpo salvo: '../datasets/porto_imoveis_dataset.csv'")
print(f"✅ {len(df.columns)} features (originais + {len(df.columns) - 18} novas)")
print("🚀 Pronto para regressao_precos.py!")
