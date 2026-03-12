# Model-example

Este repositorio contiene una colección de scripts en Python, inicialmente prototipados en Google Colab, que demuestran diferentes técnicas de análisis de datos, visualización y machine learning (Clustering, SVD, Anomalías).

Cada subdirectorio dentro de `src/` corresponde a un análisis específico y cuenta con un archivo `main.py` ejecutable y su propio `README.md`.

## Estructura del Proyecto

```text
/
├── .python-version          # Indica la versión sugerida de Python para este proyecto.
├── requirements.txt         # Dependencias globales necesarias.
├── README.md                # Este archivo.
├── example_data/            # Carpeta con los archivos CSV.
└── src/
    ├── visualizacion_simple/    # Ejemplo de visualización aleatoria de datos.
    ├── clustering_laliga/       # Análisis KMeans y PCA para estadísticas de fútbol.
    ├── svd_estudiantes/         # Factorización de matrices y cálculo de SVD.
    ├── deteccion_anomalias/     # Promedios móviles para detección de picos en tiempos.
    └── clustering_3d/           # Agrupación y visualización 3D sobre rasgos de personalidad.
```

## Configuración del Entorno Local (pyenv en Windows)

Para ejecutar correctamente estos ejemplos de manera localizada y sin interferir con otras instalaciones en tu equipo, usaremos `pyenv-win`. A continuación se describe cómo configurar tu entorno:

### 1. Requisitos Previos (pyenv-win)

Si no tienes `pyenv-win` instalado, la forma más fácil es usando PowerShell:

1. Abre PowerShell como Administrador.
2. Ejecuta:
   ```powershell
   Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
   ```
3. _(Opcional)_ Si hay errores de ejecución de scripts, corre `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` y vuelve a intentarlo.
4. Reinicia tu terminal de PowerShell. Puedes comprobar la instalación con `pyenv --version`.

### 2. Instalación de Python LTS

El proyecto recomienda **Python 3.11.8** (una versión madura y muy estable para Data Science).

Desde la ruta de este repositorio, ejecuta:

```powershell
# Ver versiones disponibles
pyenv install -l | findstr 3.11

# Instalar versión específica
pyenv install 3.11.8

# Comprobar que pyenv reconoce la versión
pyenv versions
```

### 3. Activación de la Versión Local

El archivo `.python-version` en este directorio le dice a pyenv qué versión de python usar automáticamente cuando te encuentres dentro de esta carpeta. Puedes asegurarte corriendo:

```powershell
pyenv local 3.11.8
```

Al verificar la versión (`python -V`), debería mostrar `Python 3.11.8`.

### 4. Instalación de Dependencias

Con la versión correcta de Python configurada, puedes instalar las librerías base para todos los ejemplos:

```powershell
# Idealmente, crea un entorno virtual (opcional pero recomendado):
python -m venv venv
.\venv\Scripts\activate

# Instalar los paquetes
pip install -r requirements.txt
```

---

## Ejecución de los Scripts

Antes de ejecutar los modelos que requieran datos, **asegúrate de que los archivos CSV** (`laliga.csv`, `Estudiantes.csv`, `anomalias.csv` y `analisis.csv`) estén alojados en el directorio `example_data/`.  
_Nota: Los scripts apuntan a `../../example_data/nombre_archivo.csv` usando rutas relativas._

Para ejecutar un script, simplemente desplázate al directorio respectivo y ejecuta `main.py`:

```powershell
cd src/clustering_laliga
python main.py
```
