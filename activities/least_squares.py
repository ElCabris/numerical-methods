# las funciones implementadas en este archivo
# son las mismas que se implementaron en la libreria pymethos (libreria personal)


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def lagrange_poly(x_vals, y_vals):
    x = sp.Symbol('x')
    poly = 0
    n = len(x_vals)
    for i in range(n):
        xi, yi = x_vals[i], y_vals[i]
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - x_vals[j]) / (xi - x_vals[j])
        poly += yi * Li
    return sp.expand(poly)

def least_squares_line(x_vals, y_vals):
    x = sp.Symbol('x')
    n = len(x_vals)
    sum_x = sum(x_vals)
    sum_y = sum(y_vals)
    sum_x2 = sum([xi**2 for xi in x_vals])
    sum_xy = sum([xi*yi for xi, yi in zip(x_vals, y_vals)])

    a = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    b = (sum_y - a*sum_x) / n

    recta = a*x + b
    return sp.simplify(recta)

def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("El intervalo no encierra una raíz (f(a) y f(b) deben tener signos opuestos).")

    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c

ci = [100, 200, 300, 400, 500, 600, 700, 800]
ti = [205, 430, 677, 945, 1233, 1542, 1872, 2224]
x = sp.Symbol('x')

P = lagrange_poly(ci, ti)
recta = least_squares_line(ci, ti)

print("Polinomio de Lagrange:")
sp.pprint(P)
print("\nRecta por mínimos cuadrados:")
sp.pprint(recta)

P_func = sp.lambdify(x, P, 'numpy')
recta_func = sp.lambdify(x, recta, 'numpy')

x_vals = np.linspace(100, 800, 300)
plt.scatter(ci, ti, color='black', label='Datos experimentales')
plt.plot(x_vals, P_func(x_vals), 'r--', label='Polinomio de Lagrange')
plt.plot(x_vals, recta_func(x_vals), 'b-', label='Recta Mínimos Cuadrados')
plt.xlabel('Contador (ci)')
plt.ylabel('Tiempo de uso (ti)')
plt.legend()
plt.title('Ajuste de datos: Lagrange y Mínimos Cuadrados')
plt.grid(True)
plt.show()


f_bis = sp.lambdify(x, P - 1000, 'numpy')


a, b = 400, 500
root = bisection_method(f_bis, a, b)
print(f"\nContador para t = 1000 (método de bisección): {root:.4f}")
