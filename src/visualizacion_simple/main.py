import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Generando datos de demostración...")
    ys = 200 + np.random.randn(100)
    x = [x for x in range(len(ys))]

    fig = plt.figure(figsize=(6, 4), facecolor='w')
    plt.plot(x, ys, '-')
    plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
    plt.title("Sample Visualization", fontsize=12)

    # Guardar el gráfico generado para el README
    plt.savefig('referencia.png')
    print("Gráfico guardado como 'referencia.png'. Mostrando en pantalla...")
    
    # Mostrar el gráfico de manera local
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
