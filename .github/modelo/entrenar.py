# train_model.py - Ingeniero de Datos y ML

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import joblib
import json

# ── 1. CARGAR Y FILTRAR EL DATASET ──────────────────────────────────────────
print("Cargando dataset...")
datos = fetch_california_housing()
df = pd.DataFrame(datos.data, columns=datos.feature_names)
df["PRECIO"] = datos.target  # precio en cientos de miles de dólares

# Nos quedamos SOLO con las 3 variables acordadas con el equipo
df_filtrado = df[["MedInc", "HouseAge", "AveRooms", "PRECIO"]]
print(df_filtrado.head())
print(f"\nForma del dataset: {df_filtrado.shape}")

# ── 2. SEPARAR VARIABLES DE ENTRADA Y SALIDA ─────────────────────────────────
X = df_filtrado[["MedInc", "HouseAge", "AveRooms"]]
y = df_filtrado["PRECIO"]

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nDatos de entrenamiento: {X_train.shape[0]} filas")
print(f"Datos de prueba:        {X_test.shape[0]} filas")

# ── 3. ENTRENAR EL MODELO CON HIPERPARÁMETRO OPTIMIZADO ──────────────────────
# Probamos 3 valores de max_depth y elegimos el mejor
print("\nOptimizando hiperparámetro max_depth...")
resultados = {}
for profundidad in [3, 5, 10]:
    modelo_temp = RandomForestRegressor(max_depth=profundidad, random_state=42)
    modelo_temp.fit(X_train, y_train)
    puntaje = r2_score(y_test, modelo_temp.predict(X_test))
    resultados[profundidad] = puntaje
    print(f"  max_depth={profundidad}  →  R² = {puntaje:.4f}")

mejor_profundidad = max(resultados, key=resultados.get)
print(f"\n✅ Mejor max_depth: {mejor_profundidad} (R² = {resultados[mejor_profundidad]:.4f})")

# ── 4. ENTRENAR EL MODELO FINAL CON EL MEJOR HIPERPARÁMETRO ──────────────────
modelo_final = RandomForestRegressor(max_depth=mejor_profundidad, random_state=42)
modelo_final.fit(X_train, y_train)

# ── 5. CALCULAR MÉTRICAS FINALES ──────────────────────────────────────────────
y_pred = modelo_final.predict(X_test)
r2    = r2_score(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)

print(f"\n📊 Métricas finales:")
print(f"   R²   = {r2:.4f}")
print(f"   MSE  = {mse:.4f}")
print(f"   RMSE = {rmse:.4f}")

# ── 6. EXPORTAR EL MODELO ─────────────────────────────────────────────────────
joblib.dump(modelo_final, "modelo_casas.pkl")
print("\n💾 Modelo guardado como: modelo_casas.pkl")

# ── 7. EXPORTAR LAS MÉTRICAS EN JSON (para el Integrante 4) ──────────────────
metricas = {
    "r2_score": round(r2, 4),
    "mse":      round(mse, 4),
    "rmse":     round(rmse, 4),
    "mejor_parametro": {"max_depth": mejor_profundidad},
    "variables_usadas": ["MedInc", "HouseAge", "AveRooms"]
}

with open("metricas_modelo.json", "w") as f:
    json.dump(metricas, f, indent=4)

print("📄 Métricas guardadas en: metricas_modelo.json")
print("\n✅ ¡Proceso completado! Entrega al equipo:")
print("   → modelo_casas.pkl")
print("   → metricas_modelo.json")
