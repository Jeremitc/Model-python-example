# Clustering 3D de Personalidades

Este script de análisis estadístico aplica el algoritmo **K-Means** pero enfocado en un modelado de 3 dimensiones:

- `op` (Responsabilidad/Apertura).
- `ex` (Extraversión).
- `ag` (Amabilidad/Empatía).

El programa carga datos tabulares desde `../../example_data/analisis.csv`, muestra las correlaciones 2D a través de pairplots de `Seaborn`, define un "gráfico de codo" (Elbow curve) para hallar el número ideal de clusters y luego despliega una previsualización de la distribución 3D utilizando `mpl_toolkits.mplot3d` en Matplotlib. Finalmente, el script halla y clasifica a qué cluster corresponde un usuario particular basándose en las tres métricas continuas.

## Ejecución

Primero, asegúrate de que el entorno esté activo y tengan todos los requisitos y luego corre:

```bash
python main.py
```

## Ejemplo 3D

![Clustering 3D K-Means](referencia.png)
