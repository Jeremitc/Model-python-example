import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Configuración inicial
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

import os

def main():
    dataset_path = "../../example_data/analisis.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        return

    dataframe = pd.read_csv(dataset_path)
    print("Resumen de las primeras filas:")
    print(dataframe.head())
    print("\nDescripción estadística:")
    print(dataframe.describe())
    print("\nTamaño por categorías actuales:")
    print(dataframe.groupby('categoria').size())

    # Histogramas
    print("\nGenerando histogramas (cierra la ventana para continuar)...")
    dataframe.drop(['categoria'], axis=1).hist()
    plt.show()

    # Pairplot
    print("\nGenerando pairplot (cierra la ventana para continuar)...")
    # Cambiamos 'size' (obsoleto) por 'height'
    sb.pairplot(dataframe.dropna(), hue='categoria', height=4, vars=["op", "ex", "ag"], kind='scatter')
    plt.show()

    # Preparar datos 3D
    X = np.array(dataframe[["op", "ex", "ag"]])
    y = np.array(dataframe['categoria'])
    
    # Visualización 3D Original
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colores = ['blue', 'red', 'green', 'blue', 'cyan', 'yellow', 'orange', 'black', 'pink', 'brown', 'purple']
    asignar = [colores[row] for row in y]

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60)
    plt.title("Visualización 3D Original")
    plt.show()

    # Curva del Codo (Elbow Curve)
    Nc = range(1, 20)
    kmeans_models = [KMeans(n_clusters=i, random_state=42) for i in Nc]
    score = [kmeans_models[i].fit(X).score(X) for i in range(len(kmeans_models))]
    
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()

    # Ejecutar K-Means con 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    centroids = kmeans.cluster_centers_
    print("Centroides K-Means (5 clusters):")
    print(centroids)

    # Predecir clusters
    labels = kmeans.predict(X)
    C = kmeans.cluster_centers_
    colores_kmeans = ['red', 'green', 'blue', 'cyan', 'yellow']
    asignar_kmeans = [colores_kmeans[row] for row in labels]

    # Visualización 3D clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar_kmeans, s=60)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores_kmeans, s=1000)
    plt.title("K-Means 3D Clusters (Categorías)")
    # Guardar este gráfico como referencia
    plt.savefig('referencia.png')
    plt.show()

    # Visualización 2D (ex vs op)
    f1 = dataframe['op'].values
    f2 = dataframe['ex'].values
    plt.figure()
    plt.scatter(f1, f2, c=asignar_kmeans, s=70)
    plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores_kmeans, s=1000)
    plt.title("K-Means 2D (Responsabilidad vs Extraversión)")
    plt.show()

    # Visualización 2D (ag vs ex)
    f1 = dataframe['ex'].values
    f2 = dataframe['ag'].values
    plt.figure()
    plt.scatter(f1, f2, c=asignar_kmeans, s=70)
    plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores_kmeans, s=1000)
    plt.title("K-Means 2D (Extraversión vs Amabilidad)")
    plt.show()

    # Resumen y Diversidad
    copy = pd.DataFrame()
    copy['usuario'] = dataframe['usuario'].values
    copy['categoria'] = dataframe['categoria'].values
    copy['label'] = labels

    cantidadGrupo = pd.DataFrame()
    cantidadGrupo['color'] = colores_kmeans
    cantidadGrupo['cantidad'] = copy.groupby('label').size()
    print("\nCantidad de elementos por grupo K-Means:")
    print(cantidadGrupo)

    group_referrer_index = copy['label'] == 0
    group_referrals = copy[group_referrer_index]
    diversidadGrupo = pd.DataFrame()
    diversidadGrupo['categoria'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    diversidadGrupo['cantidad'] = group_referrals.groupby('categoria').size()
    print("\nDiversidad en el Grupo 0:")
    print(diversidadGrupo)

    # Representante cercano al centroide
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    print("\nUsuarios más cercanos al centro de cada grupo:")
    users = dataframe['usuario'].values
    for row in closest:
        print(users[row])

    # Predicción nueva
    print("\nPredicción para X_new = [[45.92, 57.74, 15.66]] (ej. David Guetta):")
    X_new = np.array([[45.92, 57.74, 15.66]])
    new_labels = kmeans.predict(X_new)
    print("Cluster asignado:", new_labels[0])

if __name__ == "__main__":
    main()
