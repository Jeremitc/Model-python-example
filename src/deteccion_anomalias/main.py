import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    dataset_path = "../../example_data/anomalias.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} no encontrado.")
        return

    # Cargamos datos usando primera columna como índice (asumimos que es tiempo u otro identificador)
    data = pd.read_csv(dataset_path, index_col=1)
    
    # Opcional: limpiar data drop para evitar datos corruptos
    # data = data.dropna()

    print("Primeras filas del dataset:")
    print(data.head())

    # Visualización 1: Gráfico original
    plt.figure(figsize=(12, 6))
    data["count"].plot(title="Datos Originales")
    plt.show()

    # Promedios móviles
    wind = 15
    sigma = 2

    # Límites suelo y techo basados en Std y Promedio
    data["suelo"] = data["count"].rolling(window=wind).mean() - (sigma * data["count"].rolling(window=wind).std())
    data["techo"] = data["count"].rolling(window=wind).mean() + (sigma * data["count"].rolling(window=wind).std())

    # Visualización 2: Gráfico con límites de tolerancia
    plt.figure(figsize=(12, 6))
    data[["count", "suelo", "techo"]].plot(title="Promedios móviles y márgenes (Suelo/Techo)")
    plt.show()

    # Clasificación de anomalías (1 o 0)
    data["anom"] = data.apply(
        lambda row: row["count"] if (row["count"] <= row["suelo"] or row["count"] >= row["techo"]) else 0,
        axis=1
    )

    # Visualización 3: Destacar anomalías
    plt.figure(figsize=(12, 6))
    ax = data["count"].plot(label='Count', color='blue', alpha=0.5)
    
    # Resaltar en rojo los puntos anómalos
    anomalies = data[data["anom"] != 0]
    ax.scatter(anomalies.index, anomalies["anom"], color='red', label='Anomalía', zorder=5)

    plt.title("Detección Final de Anomalías")
    plt.legend()
    plt.savefig('referencia.png')
    plt.show()

if __name__ == "__main__":
    main()
