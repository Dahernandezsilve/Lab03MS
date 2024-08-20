# -----------------------------------------------------------------------------
# Universidad del Valle de Guatemala 
# 18 Avenida 11-95 Guatemala
# Facultad de Ingeniería
# Departamento de Computación
#
# Laboratorio 3. Modelación y Simulación 
#
# Grupo: 6
#
# Integrantes:
#   Diego Alexander Hernández Silvestre, 21270
#   Mario Antonio Guerra Morales, 21008
#   Linda Inés Jiménez Vides, 21169
#
# Curso: Modelación y Simulación
# Sección: 10
#
# Guatemala, 20 de agosto de 2024
# -----------------------------------------------------------------------------

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def sistemaEDO(X, Y):
    dx_dt = 0.2 * X - 0.005 * X * Y
    dy_dt = -0.5 * Y + 0.01 * X * Y
    return dx_dt, dy_dt


def graphCampoVectorial(showPlot=False):
    x = np.linspace(0, 150, 20)
    y = np.linspace(0, 150, 20)
    X, Y = np.meshgrid(x, y)
    U, V = sistemaEDO(X, Y)
    plt.quiver(X, Y, U, V, color='r')
    plt.title('[Inciso a] - Campo Vectorial del Sistema de EDO')
    plt.xlabel('Población x(t)')
    plt.ylabel('Población y(t)')
    plt.xlim([0, 150])
    plt.ylim([0, 150])
    plt.grid()

    if showPlot:
        plt.show()


def puntosDeEquilibrio():
    x, y = sp.symbols('x y')
    dx_dt, dy_dt = sistemaEDO(x, y)
    eq1 = sp.Eq(dx_dt, 0)
    eq2 = sp.Eq(dy_dt, 0)
    soluciones = sp.solve([eq1, eq2], (x, y))
    print(f"☝️  Puntos de equilibrio: {soluciones}")


def rungeKutta4Simple(f, y0, t0, tf, h, printR=True):
    n = int((tf - t0) / h) 
    t = t0  
    y = y0
    resultados = [y] 

    print("🚀 Iniciando Runge-Kutta [orden 4] para sistemas de EDO...")
    
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
        t = t + h  
        
        if isinstance(y, np.ndarray):
            y_str = np.array2string(y, precision=4, separator=',')
        else:
            y_str = f"{y:.4f}"

        if printR:
            print(f"⏱️  Paso {i+1}/{n}: t = {t:.4f}, y(t) = {y_str}")
        
        resultados.append(y)

    print("🏁 Finalizado!")
    return np.array(resultados) 


def sistemaEDOIncisoC(t, estado):
    X, Y = estado
    dx_dt = 0.2 * X - 0.005 * X * Y
    dy_dt = -0.5 * Y + 0.01 * X * Y
    return np.array([dx_dt, dy_dt])


def rungeKuttaIncisoC(x0=70, y0=30, printR=True):
    estadoInicial = np.array([x0, y0])
    t0 = 0  
    tf = 60  
    h = 0.1 

    resultado = rungeKutta4Simple(sistemaEDOIncisoC, estadoInicial, t0, tf, h, printR=printR)

    x_sol = resultado[:, 0]  # Primera columna es x(t)
    y_sol = resultado[:, 1]  # Segunda columna es y(t)
    t_sol = np.arange(t0, tf + h, h)

    plt.figure(figsize=(10, 5))
    plt.plot(t_sol, x_sol, label='Población x(t)')
    plt.plot(t_sol, y_sol, label='Población y(t)')
    plt.title('[Inciso c] - Solución del sistema de EDO con Runge-Kutta (x(0) = 70, y(0) = 30)')
    plt.xlabel('Tiempo (meses)')
    plt.ylabel('Población')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"☝️  Población de x(t) después de 5 años: {x_sol[-1]:.2f}")
    print(f"☝️  Población de y(t) después de 5 años: {y_sol[-1]:.2f}")


def graphTrayectoriasFaseXY():
    resultado1 = rungeKutta4Simple(sistemaEDOIncisoC, np.array([70, 30]), 0, 60, 0.1, printR=False)
    x_sol1 = resultado1[:, 0]
    y_sol1 = resultado1[:, 1]
    resultado2 = rungeKutta4Simple(sistemaEDOIncisoC, np.array([100, 10]), 0, 60, 0.1, printR=False)
    x_sol2 = resultado2[:, 0]
    y_sol2 = resultado2[:, 1]

    graphCampoVectorial(showPlot=False)

    plt.plot(x_sol1, y_sol1, label='Trayectoria (x0=70, y0=30)', color='b')
    plt.plot(x_sol2, y_sol2, label='Trayectoria (x0=100, y0=10)', color='g')
    plt.scatter([70], [30], color='blue', marker='o', s=100, label='Inicio (70,30)')
    plt.scatter([100], [10], color='green', marker='o', s=100, label='Inicio (100,10)')
    plt.scatter([x_sol1[-1]], [y_sol1[-1]], color='blue', marker='x', s=100, label=f'Final (x={x_sol1[-1]:.2f}, y={y_sol1[-1]:.2f})')
    plt.scatter([x_sol2[-1]], [y_sol2[-1]], color='green', marker='x', s=100, label=f'Final (x={x_sol2[-1]:.2f}, y={y_sol2[-1]:.2f})')
    plt.title('[Inciso e] - Trayectorias en el plano de fase xy sobre el campo vectorial')
    plt.xlabel('Población x(t)')
    plt.ylabel('Población y(t)')
    plt.xlim([0, 150])
    plt.ylim([0, 150])
    plt.legend()
    plt.grid()
    plt.show()


# Inciso a)
#print("\nInciso a) - Campo Vectorial del Sistema de EDO")
#graphCampoVectorial(showPlot=True)

# Inciso b)
#print("\nInciso b) - Puntos de Equilibrio")
#puntosDeEquilibrio()

# Inciso c)
#print("\nInciso c) - Solución del sistema de EDO con Runge-Kutta (x(0) = 70, y(0) = 30)")
#rungeKuttaIncisoC(printR=False)

# Inciso d)
#print("\nInciso d) - Solución del sistema de EDO con Runge-Kutta (x(0) = 100, y(0) = 10)")
#rungeKuttaIncisoC(x0=100, y0=10, printR=False)

# Inciso e)
print("\nInciso e) - Trayectorias en el plano de fase xy sobre el campo vectorial")
graphTrayectoriasFaseXY()