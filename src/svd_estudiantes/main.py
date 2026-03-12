import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

def main():
    dataset_path = '../../example_data/Estudiantes.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        print("Por favor, asegúrate de colocar el archivo 'Estudiantes.csv' en la carpeta 'example_data/'.")
        return

    estudiantes = pd.read_csv(dataset_path, index_col=0)
    
    # Factorización de la Matriz Estudiantes
    U, s, V_transp = svd(estudiantes)

    print("Matriz U:\n", U)
    print("Valores Singulares:\n", s)

    sigma_s = [(val / sum(s)) for val in s]
    print('Información que aporta cada valor singular:\n', sigma_s)

    plt.figure(figsize=(4, 4))
    plt.plot(range(1, len(s)+1), sigma_s)
    plt.title("Proporción de Varianza de Valores Singulares")
    plt.show()

    print('Porcentaje acumulado:', sum(sigma_s[0:2]) * 100, '% de la información contenida')

    m, n = np.shape(estudiantes)
    print('Filas:', m, '- Columnas:', n)

    sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma[i, i] = s[i]

    n_valores = 2
    U_k = U[:, :n_valores]
    sigma_k = sigma[:n_valores, :n_valores]
    VT_k = V_transp[:n_valores, :]

    A = U_k.dot(sigma_k).dot(VT_k)
    print("Matriz Reconstruida (A):\n", A)

    estudiantes_array = np.array(estudiantes)
    error = np.mean(np.abs((estudiantes_array - A) / estudiantes_array)) * 100
    print("Error Porcentual Absoluto Medio:", error)

    # Graficar resultados (SVD 1 vs SVD 2)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('SVD 1', fontsize=12)
    ax.set_ylabel('SVD 2', fontsize=12)
    ax.set_title('SVD con Matriz Reducida', fontsize=14)
    ax.scatter(x=A[:, 0], y=A[:, 1])
    # Ajustar heurísticamente los límites o simplemente dejarlos automáticos
    ax.autoscale()

    nombres = estudiantes.index
    for i, nombre in enumerate(nombres):
        ax.annotate(nombre, (A[i, 0], A[i, 1]), fontsize=8)

    plt.savefig('referencia.png')
    plt.show()

    B = np.dot(U_k, sigma_k)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('SVD 1', fontsize=12)
    ax.set_ylabel('SVD 2', fontsize=12)
    ax.set_title('Vectores Singulares Izquierdos', fontsize=14)
    ax.scatter(x=B[:, 0], y=B[:, 1])
    ax.autoscale()

    for i, nombre in enumerate(nombres):
        ax.annotate(nombre, (B[i, 0], B[i, 1]), fontsize=8)

    plt.show()

    # Estandarización
    escala_std = StandardScaler()
    estudiantes_std = escala_std.fit_transform(estudiantes)
    
    # SVD de estandarizada
    U_std, s_std, V_transp_std = svd(estudiantes_std)
    sigma_std = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma_std[i, i] = s_std[i]
        
    U_k_std = U_std[:, :2]
    sigma_k_std = sigma_std[:2, :2]
    B_std = np.dot(U_k_std, sigma_k_std)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('SVD 1', fontsize=12)
    ax.set_ylabel('SVD 2', fontsize=12)
    ax.set_title('SVD - Datos Estandarizados', fontsize=14)
    ax.scatter(x=B_std[:, 0], y=B_std[:, 1])
    ax.autoscale()

    for i, nombre in enumerate(nombres):
        ax.annotate(nombre, (B_std[i, 0], B_std[i, 1]), fontsize=8)

    plt.show()

if __name__ == "__main__":
    main()
