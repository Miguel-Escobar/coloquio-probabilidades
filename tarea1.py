import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(tf, dt, D, t0=0, n_realizations=1):
    t = np.arange(t0, tf, dt)
    x = np.zeros((len(t), n_realizations))
    x[0, :] = np.random.normal(0, D*t0, size=n_realizations)
    x[1:,:] = np.cumsum(np.sqrt(2*D*dt)*np.random.normal(size=(len(t)-1, n_realizations)), axis=0) + x[0]
    return t, x

def single_brownian(tf=1, dt=0.001, D=1):
    t, x = brownian_motion(tf, dt, D)
    plt.clf()
    plt.plot(t, x[:, 0])
    plt.xlabel('t')
    plt.ylabel('$B_t$')
    plt.title('Movimiento Browniano')
    plt.show()
    input("Presione Enter para continuar...\n")
    plt.close("all")
    return

def multiple_brownian(n, tf=100, dt=0.001, D=1, hist=True, n_hist=5):
    fig_brownians = plt.figure()
    n_display = min(n, 100)
    ax = fig_brownians.add_subplot(111)
    ax.set_xlabel('t')
    ax.set_ylabel('$B_t$')
    ax.set_title(f'{n_display} Movimientos Brownianos')
    t, x = brownian_motion(tf, dt, D, n_realizations=n)
    ax.plot(t, x[:, :n_display])
    if hist:
        fig_histograms = plt.figure(figsize=(12, 6))
        steps = len(t)//n_hist
        for i in range(1,n_hist+1):
            ax = fig_histograms.add_subplot(n_hist,1,i)
            ax.set_xlabel('$B_{%5.3f}$' % t[(i-1)*steps])
            ax.set_ylabel('Frecuencia')
            heights, _, _ = ax.hist(x[(i-1)*steps, :], bins=20, density=True, range=(np.min(x), np.max(x)), label="Histograma simulado")
            hist_height = np.max(heights)
            ax.set_ylim(0, hist_height*1.1)
            nonzero_t = max(t[(i-1)*steps], 1e-6)
            theoretical_x = np.sort(np.append(np.linspace(np.min(x), np.max(x), 500), 0))
            ax.plot(theoretical_x, 1/np.sqrt(4*np.pi*D*nonzero_t)*np.exp(-theoretical_x**2/(4*D*nonzero_t)), label="Distribución teórica")
            ax.legend()
        fig_histograms.tight_layout()
    plt.show()
    input("Presione Enter para continuar...\n")
    plt.close("all")
    return

def move_to_box(x, L):
    return (x-L*0.5) % L - L*0.5

def periodic_theoretical_distribution(x, t, D, L, n_trunc = 20):
    x_m_array = np.stack([x + m*L for m in range(-n_trunc, n_trunc+1)])
    return np.sum(1/np.sqrt(4*np.pi*D*t)*np.exp(-x_m_array**2/(4*D*t)), axis=0)
    
def periodic_brownians(n, tf=10, dt=0.001, D=1, L=10, hist=True, n_hist=5):
    fig_brownians = plt.figure()
    n_display = min(n, 100)
    ax = fig_brownians.add_subplot(111)
    ax.set_xlabel('t')
    ax.set_ylabel('$B_t$')
    ax.set_title(f'{n_display} Movimientos Brownianos')
    t, x = brownian_motion(tf, dt, D, n_realizations=n)
    x = move_to_box(x, L)
    ax.plot(t, x[:, :n_display])
    ax.set_ylim(-L, L)

    if hist:
        fig_histograms = plt.figure(figsize=(12, 6))
        steps = len(t)//n_hist
        for i in range(1,n_hist+1):
            ax = fig_histograms.add_subplot(n_hist,1,i)
            ax.set_xlabel('$B_{%5.0f}$' % t[(i-1)*steps])
            ax.set_ylabel('Frecuencia')
            heights, _, _ = ax.hist(x[(i-1)*steps, :], bins=19, density=True, range=(-L*0.5, L*0.5), label="Histograma simulado")
            hist_height = np.max(heights)
            ax.set_ylim(0, hist_height*1.1)
            nonzero_t = max(t[(i-1)*steps], 1e-3)
            theoretical_x = np.linspace(-L*0.5, L*0.5, 500)
            ax.plot(theoretical_x, periodic_theoretical_distribution(theoretical_x, nonzero_t, D, L), label="Distribución teórica")
            ax.legend()
        fig_histograms.tight_layout()
    plt.show()
    input("Presione Enter para continuar...\n")
    plt.close("all")
    return

if __name__ == '__main__':
    L = 10
    tf = 10
    dt = 0.001
    D = 1
    print("""¡Bienvenido al programa de simulación de movimientos brownianos!
Intenté hacer algo un poco más interactivo (un poco).
En caso de que surja cualquier problema al correr este programa, por favor
intentar correrlo desde ipython --matplotlib (uno nunca sabe si funciona en pc ajeno).\n""")
    print("""Primero, observemos un sólo movimiento browniano.
Se abrirá una ventana con el gráfico de B_t en función de t.\n""")
    single_brownian(tf, dt, D)
    print("""Ahora, observemos varios movimientos brownianos.
Se abrirá una ventana con el gráfico de B_t en función de t.
Además, se abrirá una ventana con histogramas de B_t en distintos tiempos.
Note cómo se puede observar que los histogramas se ajustan a la distribución teórica.
Esto corresponde a la condición de borde de anularse en infinito.\n""")
    multiple_brownian(5000, tf, dt, D)
    print(f"""Finalmente, observemos varios movimientos brownianos en un espacio periódico.
Se abrirá una ventana con el gráfico de B_t en función de t, pero mucho
más ruidosa debido a tener brownianos concentrados en un rango de {-L/2} a {L/2}.
Además, se abrirá una ventana con histogramas de B_t en distintos tiempos.
Para calcular numéricamente la distribución teórica, se consideraron 41 términos
de la serie encontrada. ¡Vemos un buen ajuste!\n""")
    periodic_brownians(5000, tf, dt, D, L)

    print("""Un último detalle: Notemos la importancia de considerar un número grande
de realizaciones para obtener una buena aproximación. Vemos que con tán solo 10 realizaciones
por ejemplo, no se logra apreciar la distribución teórica en los histogramas.\n""")
    multiple_brownian(10, tf, dt, D)
    periodic_brownians(10, tf, dt, D, L)

    print("Fin del programa. ¡Disculpe la entrega atrasada!")
    input("Presione Enter para cerrar...\n")

