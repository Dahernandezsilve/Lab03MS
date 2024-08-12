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
