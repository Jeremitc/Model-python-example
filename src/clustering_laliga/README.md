# Clustering de Jugadores de La Liga

Este repositorio incluye un script robusto para analizar datos estadísticos de jugadores de fútbol de La Liga. Originalmente estructurado en Google Colab para ser utilizado con Google Drive, el script ha sido adaptado para leer datos desde `../data/laliga.csv`.

El flujo de trabajo es:

1. Limpieza de las cabeceras/columnas.
2. Correlación de variables (Generación de matriz de calor).
3. Pre-procesamiento: `MinMaxScaler` para estandarizar métricas.
4. Aplicación de **PCA (Principal Component Analysis)** para reducir la dimensionalidad de las características numéricas en 2 componentes principales (2D).
5. Entrenamiento de un modelo de **K-Means Clustering** con $k=4$ (Porteros, Defensas, Centrales, Delanteros).
6. Visualización de la proyección de todos los jugadores en un plano 2D con etiquetas.

## Requisitos Previos

Asegúrate de haber instalado las dependencias pertinentes definidas en `../../requirements.txt` y descargar o colocar el archivo `laliga.csv` dentro de la carpeta `../../example_data/`.

## Ejecución

```bash
python main.py
```

## Ejemplo Visual del Clustering

![Proyección PCA K-Means](referencia.png)
