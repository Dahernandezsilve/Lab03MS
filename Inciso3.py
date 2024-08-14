import numpy as np
import matplotlib.pyplot as plt

# Definición del sistema de EDO
def f(t, x, y):
    dx = 0.5*x - 0.001*x**2 - x*y
    dy = -0.2*y + 0.1*x*y
    return dx, dy

# Rango de valores para x e y
x = np.linspace(-20, 20, 20)
y = np.linspace(-20, 20, 20)

# Crear una malla de puntos
X, Y = np.meshgrid(x, y)

# Calcular los campos vectoriales
U, V = f(0, X, Y)

# Obtener puntos de equilibrio
def jacobian(x, y):
    dxdx = 0.5 - 0.002*x - y
    dxdy = -x
    dydx = 0.1*y
    dydy = -0.2 + 0.1*x
    return np.array([[dxdx, dxdy], [dydx, dydy]])

def find_equilibrium(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    for i in range(max_iter):
        fx, fy = f(0, x, y)
        J = jacobian(x, y)
        dx, dy = np.linalg.solve(J, -np.array([fx, fy]))
        x, y = x + dx, y + dy
        if np.abs(fx) < tol and np.abs(fy) < tol:
            return x, y
    raise ValueError("No hay convergencia")

# Encontrar los puntos de equilibrio en el primer cuadrante
equilibrium_points = []
for x0 in np.linspace(0, 20, 10):
    for y0 in np.linspace(0, 20, 10):
        if x0 >= 0 and y0 >= 0:
            try:
                x, y = find_equilibrium(x0, y0)
                equilibrium_points.append((x, y))
            except ValueError:
                pass

def rungeKutta4SimpleInciso3(f, y0, t0, tf, h):
    n = int((tf - t0) / h) 
    t = t0  
    x = y0[0]
    y = y0[1]

    t_values = []
    x_values = []
    y_values = []

    print("🚀 Iniciando Runge-Kutta [orden 4] para EDO simple...")
    
    for i in range(n):
        k1x, k1y = f(t, x, y)
        k2x, k2y = f(t + 0.5 * h, x + 0.5 * h * k1x, y + 0.5 * h * k1y)
        k3x, k3y = f(t + 0.5 * h, x + 0.5 * h * k2x, y + 0.5 * h * k2y)
        k4x, k4y = f(t + h, x + h * k3x, y + h * k3y)

        x = x + h * (1/6 * k1x + 1/3 * k2x + 1/3 * k3x + 1/6 * k4x)
        y = y + h * (1/6 * k1y + 1/3 * k2y + 1/3 * k3y + 1/6 * k4y)
        t = t + h

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)

        print(f"⏱️ Paso {i+1}/{n}: t = {t:.4f}, x(t) = {x:.4f}, y(t) = {y:.4f}")
    
    print("🏁 Finalizado!")
    return [x, y], t_values, x_values, y_values

# Inciso A
# Graficar el campo vectorial
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Campo vectorial del sistema de EDO')
plt.grid()
plt.show()

# Inciso B

unique_points = set()
for x, y in equilibrium_points:
    unique_points.add((round(x, 4), round(y, 4)))

print("Puntos de equilibrio únicos en el primer cuadrante:")
for x, y in unique_points:
    print(f"(x, y) = ({x}, {y})")

# Clasifica el comportamiento de los puntos de equilibrio únicos
for x, y in unique_points:
    J = jacobian(x, y)
    eigenvalues, _ = np.linalg.eig(J)
    if all(eigenvalues < 0):
        print(f"El punto de equilibrio ({x}, {y}) es un nodo estable.")
    elif all(eigenvalues > 0):
        print(f"El punto de equilibrio ({x}, {y}) es un nodo inestable.")
    else:
        print(f"El punto de equilibrio ({x}, {y}) es un punto de silla.")

# Inciso C

# Condiciones iniciales
x0 = 10
y0 = 10
t0 = 0
tf = 60  # 5 años, asumiendo 1 mes = 1 unidad de tiempo
h = 0.4  # Tamaño del paso

# Resolver el sistema de EDO
resultado, tiempo, resultadosX, resultadosY = rungeKutta4SimpleInciso3(f, y0=[x0, y0], t0=t0, tf=tf, h=h)

# Imprimir los resultados finales
print(f"Población x después de 5 años: {resultado[0]:.4f}")
print(f"Población y después de 5 años: {resultado[1]:.4f}")

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar la recta de x0 y resultado[0]
ax.plot(tiempo, resultadosX, label='Recta x')

# Graficar la recta de y0 y resultado[1]
ax.plot(tiempo, resultadosY, label='Recta y')

# Agregar etiquetas y título
ax.set_xlabel('Tiempo')
ax.set_ylabel('Población')
ax.set_title('Gráfico de Runge-Kutta Orden 4')
ax.legend()

# Mostrar el gráfico
plt.show()

# Mostrar el gráfico
plt.show()

# Inciso D
# Graficar el plano de fase y la trayectoria
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='b')
plt.plot(resultado[0], resultado[1], 'r-', label='Trayectoria')
plt.scatter(x0, y0, color='g', label='Condición inicial')
plt.scatter(resultado[0], resultado[1], color='k', label='Población final')
for x, y in unique_points:
    plt.scatter(x, y, color='y', marker='x', label='Punto de equilibrio')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plano de fase del sistema de EDO')
plt.legend()
plt.grid()
plt.show()