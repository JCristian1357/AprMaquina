# entrenar_vinos.py - Ingeniero de Datos y ML (Modelado No Supervisado)

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import joblib
import json
import os

# ── 1. CARGAR Y FILTRAR EL DATASET ──────────────────────────────────────────
print("Cargando dataset de vinos...")
datos = load_wine()
df = pd.DataFrame(datos.data, columns=datos.feature_names)

# Nos quedamos SOLO con las 4 variables acordadas con el equipo
df_filtrado = df[["alcohol", "malic_acid", "color_intensity", "flavanoids"]]
print(df_filtrado.head())
print(f"\nForma del dataset: {df_filtrado.shape}")

X = df_filtrado[["alcohol", "malic_acid", "color_intensity", "flavanoids"]]

# ── 2. ESCALAR LOS DATOS ─────────────────────────────────────────────────────
print("\nEscalando los datos con StandardScaler...")
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)
print("Datos escalados correctamente.")

# ── 3. OPTIMIZAR EL HIPERPARÁMETRO n_clusters ────────────────────────────────
# Probamos varios valores de k y elegimos el que da mejor Silhouette Score.
# Esto justifica con datos por qué usamos 3 clusters (y no un número arbitrario).
print("\nOptimizando hiperparámetro n_clusters...")
resultados = {}
for k in range(2, 9):
    modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas_temp = modelo_temp.fit_predict(X_escalado)
    sil_temp = silhouette_score(X_escalado, etiquetas_temp)
    resultados[k] = sil_temp
    print(f"  n_clusters={k}  ->  Silhouette = {sil_temp:.4f}")

mejor_k = max(resultados, key=resultados.get)
print(f"\nMejor n_clusters: {mejor_k} (Silhouette = {resultados[mejor_k]:.4f})")

# ── 4. ENTRENAR EL MODELO FINAL CON EL MEJOR HIPERPARÁMETRO ──────────────────
modelo_kmeans = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
etiquetas = modelo_kmeans.fit_predict(X_escalado)

sil_score = silhouette_score(X_escalado, etiquetas)
inercia = modelo_kmeans.inertia_

print(f"\nMetricas finales del modelo:")
print(f"   Silhouette Score = {sil_score:.4f}")
print(f"   Inercia          = {inercia:.4f}")

# Distribución de vinos por cluster (útil para verificar que no quedó un
# cluster vacío o desbalanceado de forma extraña)
distribucion = pd.Series(etiquetas).value_counts().sort_index()
print(f"\nDistribución de vinos por cluster:")
for cluster, cantidad in distribucion.items():
    print(f"   Cluster {cluster}: {cantidad} vinos")

# ── 5. EXPORTAR MODELO Y ESCALADOR ───────────────────────────────────────────
os.makedirs("modelo", exist_ok=True)
joblib.dump(modelo_kmeans, "modelo/modelo_vinos.pkl")
joblib.dump(escalador, "modelo/scaler_vinos.pkl")
print("\n💾 Modelo guardado como:    modelo/modelo_vinos.pkl")
print("💾 Escalador guardado como: modelo/scaler_vinos.pkl")

# ── 6. EXPORTAR LAS MÉTRICAS EN JSON (para el Integrante 4) ──────────────────
metricas = {
    "silhouette_score": round(sil_score, 4),
    "inertia": round(inercia, 4),
    "n_clusters": mejor_k,
    "busqueda_n_clusters": {str(k): round(v, 4) for k, v in resultados.items()},
    "variables_usadas": ["alcohol", "malic_acid", "color_intensity", "flavanoids"]
}

with open("modelo/metricas_modelo.json", "w") as f:
    json.dump(metricas, f, indent=4)

print("📄 Métricas guardadas en: modelo/metricas_modelo.json")
print("\n✅ ¡Proceso completado! Entrega al equipo:")
print("   → modelo/modelo_vinos.pkl")
print("   → modelo/scaler_vinos.pkl")
print("   → modelo/metricas_modelo.json")
