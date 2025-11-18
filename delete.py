
import numpy as np
import matplotlib.pyplot as plt

# Datos
k = 200
m = 5
mu = 0.5
g = 9.80665
c = 18


# Sistema de 2 EDOs
def sistema(t, y, v):
    dy_dt = v
    dw_dt = -((c*v) + k*y) / m 
    return dy_dt, dw_dt



# Tu función de Euler
def euler_sistema(f, x0, y0, t0, tf, h):
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0] = x0
    y[0] = y0
    
    for i in range(n_steps - 1):
        dx_dt, dy_dt = f(t[i], x[i], y[i])
        x[i+1] = x[i] + h*dx_dt
        y[i+1] = y[i] + h*dy_dt
    
    return t, x, y

# Resolver con Euler
h = 0.0005
t, y, v = euler_sistema(sistema, 0.1, 0, 0, 2, h)


plt.figure(figsize=(10,4))
plt.plot(t, y)
plt.title("Posición y(t)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.grid()

plt.figure(figsize=(10,4))
plt.plot(t, v)
plt.title("Velocidad v(t)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid()

plt.show()
