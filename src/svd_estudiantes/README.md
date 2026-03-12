# SVD Factorización de Estudiantes

Este script utiliza la técnica matricial **SVD** (Descomposición en Valores Singulares), una operación fundamental en el Machine Learning para reducción de dimensionalidad, extracción de características y compresión.

A través del dataset de estudiantes (importado de forma local desde `../../example_data/Estudiantes.csv`), calculamos las matrices resultantes `U`, `Sigma`, y `V^T` tanto en sus representaciones naturales como con estandarización usando `StandardScaler`.

## Requisitos y Uso

Validar contar con las bibliotecas NumPy, SciPy, Matplotlib y Scikit-Learn.

```bash
python main.py
```

## Visualización de Componentes Principales

![Reducción SVD](referencia.png)
