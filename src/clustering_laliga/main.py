import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def main():
    # Asegurar que estamos trabajando en un directorio donde example_data/laliga.csv es accesible
    dataset_path = "../../example_data/laliga.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        print("Por favor, asegúrate de colocar el archivo 'laliga.csv' en la carpeta 'example_data/'.")
        return

    # Cargar el dataset
    players_df = pd.read_csv(dataset_path)

    # Limpiar los nombres de las columnas: eliminar espacios al principio/final y convertir a minúsculas
    players_df.columns = players_df.columns.str.strip().str.replace(' ', '_').str.lower()

    # Imprimir los nombres de las columnas después de la limpieza
    print("Nombres de columnas disponibles:", players_df.columns)

    # Mapeo de los nombres de las columnas que estás buscando con las columnas disponibles
    column_mapping = {
        'jugador': 'jugador',
        'categoria': 'categoria',
        'partidos': 'partidos',
        'minutos_jugados': 'minutos_jugados',
        'goles': 'goles',
        'asistencias': 'asistencias',
        'paradas': 'paradas',
        'goles_encajados': 'goles_encajados',
        'tiros': 'tiros',
        'tiros_a_puerta': 'tiros_a__puerta',
        'precision_tiros': 'precision_tiros',
        'centros': 'centros',
        'centros_precisos': 'centros__precisos',
        'precision_centros': 'precision_centros',
        'tiros_al_palo': 'tiros_al_palo',
        'corners_forzados': 'corners_forzados',
        'faltas_recibidas': 'faltas_recibidas',
        'faltas_cometidas': 'faltas_cometidas',
        'pases_interceptados': 'pases_interceptados',
        'balones_robados': 'balones_robados',
        'penaltis_cometidos': 'penaltis_cometidos',
        'penaltis_forzados': 'penaltis_forzados',
        'penaltis_lanzados': 'penaltis_lanzados',
        'penaltis_anotados': 'penaltis_anotados',
        'penaltis_parados': 'penaltis_parados',
        'goles_en_propia_meta': 'goles_en_propia_meta',
        'tiros_bloqueados': 'tiros_bloqueados',
        'errores_en_gol_en_contra': 'errores_en_gol_en_contra',
        'regates_con_exito': 'regates_con_exito',
        'pasesconexito': 'pasesconexito',
        'precisionpases': 'precisionpases',
        'cornerscolgados': 'cornerscolgados',
        'faltascolgadas': 'faltascolgadas',
        'faltascolgadasprecisas': 'faltascolgadasprecisas',
        'faltasdirectas': 'faltas_directas',
        'faltasdirectasapuerta': 'faltasdirectasapuerta',
        'golesdefalta': 'golesdefalta'
    }

    # Seleccionar los jugadores con más de 2000 minutos jugados
    players_df = players_df[players_df['minutos_jugados'] > 2000]
    print(f"Dimensiones del dataset tras filtrado: {players_df.shape}")

    # Usamos el mapeo para seleccionar las columnas que coinciden
    columns_to_use = list(column_mapping.values())
    
    # Filtrar solo columnas que realmente existan en el dataframe
    valid_columns = [col for col in columns_to_use if col in players_df.columns]
    players_df = players_df[valid_columns]

    # Guardar los nombres de los jugadores en una lista
    names = players_df['jugador'].tolist()

    # Eliminar las columnas categóricas para el análisis numérico
    if 'jugador' in players_df.columns and 'categoria' in players_df.columns:
        players_df = players_df.drop(['jugador', 'categoria'], axis=1)

    print("Cabecera del dataset para el análisis:")
    print(players_df.head())

    # Desplegamos una matriz de correlación
    print("Mostrando Matriz de Correlación...")
    corr = players_df.corr()
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    )
    plt.tight_layout()
    plt.show()

    # Realizamos el escalado o normalizado de los datos del dataset
    x = players_df.values  # numpy array
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)

    print("Datos normalizados:")
    print(X_norm.head())

    # Creamos gráficos antes y después de la transformación
    print("Mostrando densidades antes y después de normalizar...")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))
    ax1.set_facecolor('#E8E8F1')
    ax2.set_facecolor('#E8E8F1')
    ax1.set_title('Antes de la transformación')
    
    if 'minutos_jugados' in players_df.columns: sns.kdeplot(players_df['minutos_jugados'], ax=ax1)
    if 'goles' in players_df.columns: sns.kdeplot(players_df['goles'], ax=ax1)
    if 'tiros' in players_df.columns: sns.kdeplot(players_df['tiros'], ax=ax1)
    
    ax2.set_title('Después de la transformación')
    if len(X_norm.columns) > 8:
        sns.kdeplot(X_norm[1], ax=ax2)
        sns.kdeplot(X_norm[2], ax=ax2)
        sns.kdeplot(X_norm[8], ax=ax2)
    plt.tight_layout()
    plt.show()

    # Genera los PCA para generar el modelo
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X_norm)
    reduced = pd.DataFrame(reduced_features)

    # Especifica el número de clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans = kmeans.fit(reduced)

    # Añadimos los clusters y nombres
    clusters = kmeans.labels_.tolist()
    reduced['cluster'] = clusters
    reduced['name'] = names
    reduced.columns = ['x', 'y', 'cluster', 'name']
    
    print("Distribución PCA y Clusters:")
    print(reduced.head())
    print("\nCantidad por Cluster:")
    print(reduced.groupby('cluster').size())

    # Configuración de DataFrames de resumen según requerimiento original
    tabla = {
        'Categoria': ['1 (porteros)', '2 (defensas)', '3 (centrales)', '4 (delanteros)'],
        'Cantidad Original': [16, 33, 24, 11],
        'Cantidad K-means': [16, 30, 23, 15]
    }
    df = pd.DataFrame(tabla)
    df.index.name = 'Cluster'
    print("\nResumen comparativo:")
    print(df)

    # Visualización del resultado del clustering
    print("Mostrando proyección PCA con K-Means Clusters...")
    sns.set_theme(style="white")
    ax_lm = sns.lmplot(x="x", y="y", hue='cluster', data=reduced, legend=False, fit_reg=False, height=10, scatter_kws={"s": 150})
    for x_val, y_val, s_val in zip(reduced.x, reduced.y, reduced.name):
        plt.text(x_val, y_val, s_val, fontsize=9)
    plt.ylim(-2, 2)
    plt.tick_params(labelsize=12)
    plt.xlabel("PC 1", fontsize=15)
    plt.ylabel("PC 2", fontsize=15)
    
    # Save a reference image before showing
    plt.savefig('referencia.png')
    
    plt.show()

    # Correlación entre minutos jugados y otros
    print("Mostrando diagramas de dispersión de Goles vs Otras métricas...")
    def make_scatter(df_data):
        feats = ('minutos_jugados', 'tiros', 'centros', 'tiros')
        plt.figure(figsize=(15, 10))
        for index, feat in enumerate(feats):
            if feat in df_data.columns and 'goles' in df_data.columns:
                plt.subplot((len(feats) + 3) // 4, 4, index + 1)
                ax_reg = sns.regplot(x='goles', y=feat, data=df_data)
                ax_reg.set_title(feat)
        plt.subplots_adjust(hspace=0.4)

    make_scatter(players_df)
    plt.show()

if __name__ == "__main__":
    main()
