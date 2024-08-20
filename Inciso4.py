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
import matplotlib.pyplot as plt

def rungeKutta4SistemaEDOS(f, y0, t0, tf, h, printR=True):
    n = int((tf - t0) / h) 
    t = t0  
    y = y0
    resultados = [y] 
    tiempos = [t]

    print("🚀 Iniciando Runge-Kutta [orden 4] para sistemas de EDO...")
    posiciones = y[:3]
    velocidades = y[3:]
    pos_str = np.array2string(posiciones, precision=4, separator=',')
    vel_str = np.array2string(velocidades, precision=4, separator=',')
    print(f"🔢 Condiciones iniciales: t = {t0:.4f} años,  Posicion = {pos_str}, Velocidad = {vel_str}")

    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
        t = t + h  

        tiempos.append(t)
        resultados.append(y)
        
        if printR and (np.isclose(t, 100, atol=h/2) or np.isclose(t, 200, atol=h/2)):
            posiciones = y[:3]
            velocidades = y[3:]
            pos_str = np.array2string(posiciones, precision=4, separator=',')
            vel_str = np.array2string(velocidades, precision=4, separator=',')
            print(f"⏱️  Tiempo {t:.4f} años: Posicion = {pos_str}, Velocidad = {vel_str}")
            

    print("🏁 Simulación completada.")
    
    return np.array(resultados), np.array(tiempos)

# Definir el sistema de ecuaciones para el cometa Halley
def sistemaCometa(t, y):
    mu = 4 * np.pi**2  # Constante gravitacional en UA^3 / año^2
    x, y, z, vx, vy, vz = y

    # Calcular r (distancia al origen)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Ecuaciones de movimiento
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3
    
    # Retornar el sistema de ecuaciones de primer orden
    return np.array([vx, vy, vz, ax, ay, az])

# Condiciones iniciales
y0 = np.array([0.325514, -0.459460, 0.166229, -9.096111, -6.916868, -1.305721])
t0 = 0.0   # Tiempo inicial (1986)
tf = 200.0  # Tiempo final (años)
h = 0.01    # Tamaño del paso

# Ejecutar el método de Runge-Kutta de orden 4
resultados, tiempos = rungeKutta4SistemaEDOS(sistemaCometa, y0, t0, tf, h)

print("Resultados", resultados)

# Graficar las proyecciones xy, xz, yz
x = resultados[:, 0]
y = resultados[:, 1]
z = resultados[:, 2]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x, y, color='pink')
plt.xlabel('x (UA)')
plt.ylabel('y (UA)')
plt.title('Proyección XY')

plt.subplot(1, 3, 2)
plt.plot(x, z, color='pink')
plt.xlabel('x (UA)')
plt.ylabel('z (UA)')
plt.title('Proyección XZ')

plt.subplot(1, 3, 3)
plt.plot(y, z, color='pink')
plt.xlabel('y (UA)')
plt.ylabel('z (UA)')
plt.title('Proyección YZ')

plt.tight_layout()
plt.show()

# Gráfica de t vs r(t)
r = np.sqrt(x**2 + y**2 + z**2)
plt.figure()
plt.grid()
plt.plot(tiempos, r, color='pink')
plt.xlabel('Tiempo (años)')
plt.ylabel('Distancia r (UA)')
plt.title('Gráfica de t vs r(t)')
plt.show()
