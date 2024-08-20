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

def rungeKutta4Simple(f, y0, t0, tf, h):
    n = int((tf - t0) / h) 
    t = t0  
    y = y0 

    print("🚀 Iniciando Runge-Kutta [orden 4] para EDO simple...")
    
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
        t = t + h  
        
        print(f"⏱️ Paso {i+1}/{n}: t = {t:.4f}, y(t) = {y:.4f}")
    
    print("🏁 Finalizado!")
    return y

# Ejemplo de uso: Ley de Enfriamiento de Newton
def leyEnfriamientoNewton(t, T):
    T_amb = 20  # Temperatura ambiente (en grados Celsius)
    k = 0.1     # Constante de enfriamiento
    return -k * (T - T_amb)

# Condiciones iniciales
T0 = 100  # Temperatura inicial del objeto (en grados Celsius)
t0 = 0    # Tiempo inicial
tf = 5    # Tiempo final
h = 0.5   # Tamaño del paso

# Resolviendo la EDO
resultado = rungeKutta4Simple(leyEnfriamientoNewton, y0=T0, t0=t0, tf=tf, h=h)
print(f"Resultado final: T(tf) = {resultado:.4f} °C")


def rungeKutta4SistemaEDOS(f, y0, t0, tf, h, printR=True):
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

def lotka_volterra(t, y):
    alpha = 1.5   # Tasa de crecimiento de las presas
    beta = 1.0    # Tasa de depredación
    delta = 1.0   # Tasa de aumento de depredadores por cada presa consumida
    gamma = 3.0   # Tasa de muerte de los depredadores
    
    x, y = y[0], y[1]
    
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    
    return np.array([dxdt, dydt])

# Condiciones iniciales
y0 = np.array([10, 5])  # 10 presas, 5 depredadores
t0 = 0.0                # Tiempo inicial
tf = 15.0               # Tiempo final
h = 0.01                # Tamaño del paso

# Ejecución del método de Runge-Kutta de orden 4
resultados = rungeKutta4SistemaEDOS(lotka_volterra, y0, t0, tf, h, pri)

# Imprime los resultados
print(resultados)