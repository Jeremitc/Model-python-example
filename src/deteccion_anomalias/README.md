# Detección de Anomalías

Script que detecta comportamientos fuera de lo estándar (picos) a través de estadísticas de **Límites Móviles** (Rolling Windows).  
Calcula desviaciones estándar alrededor de una media móvil para identificar observaciones "anómalas" históricamente en el flujo de los datos `count` ubicados en `../../example_data/anomalias.csv`.

## Ejecución Lógica Automática

Si la observación se ubica por encima del valor límite máximo (Techo) o por debajo de su valor límite inferior (Suelo), se considera una `anom`, para lo cual se extrae un ScatterPlot mapeándolas sobre el gráfico normal.

## Ejecución

```bash
python main.py
```

## Ejemplo del Resultado Encontrado

![Picos de Anomalía](referencia.png)
