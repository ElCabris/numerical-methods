import math
import matplotlib.pyplot as plt
import sympy as sp
from typing import Union, Tuple, Callable
import numpy as np

Number = Union[int, float, sp.Expr]


def absolute_error(real: int | float, aprox: int | float) -> int | float:
    return abs(real - aprox)


def relative_error(real: int | float, aprox: int | float) -> float:
    return absolute_error(real, aprox) / abs(real)


def percentage_error(real: int | float, aprox: int | float) -> float:
    return relative_error(real, aprox) * 100


def significant_rounding(x: float, n: int) -> float:
    if x == 0:
        return 0.0
    d = math.floor(math.log10(abs(x)))
    factor = 10 ** (n - d - 1)
    return round(x * factor) / factor


def significant_truncate(x: float, n: int) -> float:
    if x == 0:
        return 0.0
    d = math.floor(math.log10(abs(x)))
    factor = 10 ** (n - d - 1)
    return math.trunc(x * factor) / factor


def taylor_polynomial(
    f_expr: sp.Expr, var: sp.Symbol, a: Number, n: int, k: int = 0
) -> sp.Expr:
    """
    f_expr: Symbolic function
    var: Symbolical variable
    a: Expansion point
    n: order of the Taylor polynomial
    """
    if k > n:
        return sp.Integer(0)

    deriv_k: sp.Expr = f_expr.diff(var, k).subs(var, a)
    term: sp.Expr = (deriv_k / sp.factorial(k)) * (var - a) ** k

    return term + taylor_polynomial(f_expr, var, a, n, k + 1)


def taylor_remainder_bound(
    f_expr: sp.Expr,
    var: sp.Symbol,
    a: Number,
    n: int,
    x: Number
) -> Tuple[sp.Expr, float]:
    """
    Compute the (n+1)-th derivative of a symbolic function and an upper bound
    for the Lagrange remainder of the Taylor polynomial of order `n`
    expanded at the point `a`, evaluated at the point `x`.

    This function implements the classical Lagrange remainder estimation:

        |R_n(x)| ≤ M * |x - a|^(n+1) / (n+1)!

    where:
        M = max_{c ∈ [a, x]} |f^(n+1)(c)|.

    To approximate M, the function:
        1. Computes f^(n+1)(x) symbolically.
        2. Searches for critical points of the derivative of f^(n+1).
        3. Evaluates |f^(n+1)| at the interval endpoints and at all critical points.
        4. Takes the maximum value found.

    Parameters
    ----------
    f_expr : sympy.Expr
        Symbolic expression representing the function f(x).
    var : sympy.Symbol
        The variable with respect to which derivatives are taken.
    a : Number
        The expansion point for the Taylor polynomial.
    n : int
        Order of the Taylor polynomial.
    x : Number
        The point at which the remainder bound is evaluated.

    Returns
    -------
    deriv_n1_expr : sympy.Expr
        Symbolic expression of the (n+1)-th derivative of f(x).
    bound : float
        Numerical upper bound for |R_n(x)|, the magnitude of the Lagrange remainder.

    Raises
    ------
    ValueError
        If `n < 0`.

    Notes
    -----
    - If x = a, the remainder is zero and the function returns (f^(n+1), 0).
    - If SymPy cannot solve for critical points, the function gracefully ignores them
      and uses only interval endpoints.
    - The value M is computed numerically using float evaluations of the derivative.

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> f = sp.exp(x)
    >>> deriv, bound = taylor_remainder_bound(f, x, a=0, n=3, x=1)
    >>> deriv
    exp(x)
    >>> bound  # should be close to e / 24
    0.113...
    """

    if n < 0:
        raise ValueError("Order n must be non-negative.")

    a_num = float(sp.N(a))
    x_num = float(sp.N(x))
    lo, hi = (a_num, x_num) if a_num <= x_num else (x_num, a_num)

    # If x = a, remainder is zero.
    if lo == hi:
        return sp.diff(f_expr, var, n + 1), 0.0

    # Compute (n+1)-th derivative.
    deriv_n1_expr = sp.simplify(sp.diff(f_expr, var, n + 1))

    # Attempt to find critical points of the (n+1)-th derivative.
    crit_points = []
    try:
        crit_set = sp.solveset(
            sp.Eq(sp.diff(deriv_n1_expr, var), 0),
            var,
            domain=sp.Interval(lo, hi)
        )
        if isinstance(crit_set, sp.FiniteSet):
            for s in crit_set:
                try:
                    s_val = float(sp.N(s))
                    if lo <= s_val <= hi:
                        crit_points.append(s_val)
                except Exception:
                    pass
    except Exception:
        pass

    # Evaluate absolute derivative at endpoints and critical points.
    candidates = [lo, hi] + crit_points
    max_val = 0.0
    for c in candidates:
        try:
            val = float(abs(sp.N(deriv_n1_expr.subs(var, c))))
            if val > max_val:
                max_val = val
        except Exception:
            continue

    M = max_val
    bound = M / math.factorial(n + 1) * abs(x_num - a_num) ** (n + 1)

    return deriv_n1_expr, bound


def bisection(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")

    c = (a + b) / 2
    fc = f(c)

    if abs(fc) < tol:
        return c

    if f(a) * fc < 0:
        return bisection(f, a, c, tol)
    else:
        return bisection(f, c, b, tol)


def bisection_iters(a: float, b: float, eps: float) -> int:
    if eps <= 0:
        raise ValueError("eps debe ser > 0")
    L = abs(b - a)
    if eps >= L:
        return 0
    return math.ceil(math.log(L / eps, 2))


def false_position(
    f: Callable[[float], float], a: float, b: float, tol: float
) -> float:
    """ """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")

    c = (a * fb - b * fa) / (fb - fa)
    fc = f(c)

    if abs(fc) < tol:
        return c

    if fa * fc < 0:
        return false_position(f, a, c, tol)
    else:
        return false_position(f, c, b, tol)


def newton_method(f_expr, x, x0, tol=1e-6, max_iter=100):
    df_expr = sp.diff(f_expr, x)
    x_n = x0

    for i in range(max_iter):
        f_val = f_expr.evalf(subs={x: x_n})
        df_val = df_expr.evalf(subs={x: x_n})
        if df_val == 0:
            raise ZeroDivisionError("La derivada es cero en x = {}".format(x_n))
        x_next = x_n - f_val / df_val
        if abs(x_next - x_n) < tol:
            return x_next
        x_n = x_next

    raise ValueError(f"No se encontró la raíz en {max_iter} iteraciones")


def lagrange_interpolation(xs, ys):
    if len(xs) != len(ys):
        raise ValueError()

    x = sp.symbols("x")
    n = len(xs)
    P = 0
    for i in range(n):
        # Construir el L_i(x)
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - xs[j]) / (xs[i] - xs[j])
        P += ys[i] * Li

    return sp.expand(P)


def recta_minimos_cuadrados(x_vals, y_vals):
    """
    Calcula la recta de mínimos cuadrados para un conjunto de puntos (x, y).
    Retorna la expresión simbólica y la pendiente/intercepto.
    """
    # Variable simbólica
    x = sp.Symbol("x")

    # Número de puntos
    n = len(x_vals)

    # Convertir a Sympy para operaciones exactas
    X = list(map(sp.Rational, x_vals))
    Y = list(map(sp.Rational, y_vals))

    # Calcular sumatorias
    sum_x = sum(X)
    sum_y = sum(Y)
    sum_x2 = sum(xi**2 for xi in X)
    sum_xy = sum(xi * yi for xi, yi in zip(X, Y))

    # Fórmulas de mínimos cuadrados
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - a * sum_x) / n

    # Ecuación de la recta
    recta = a * x + b

    return recta.simplify(), sp.simplify(a), sp.simplify(b)


class SLE:

    @classmethod
    def build_system(
        cls, n: int, T_bottom: float, T_left: float, T_right: float, T_top: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construye el sistema de ecuaciones lineales (A, b) para una placa cuadrada discretizada.


        Parámetros:
        ---
        n (int): Números de puntso interiores por lado (la malla será n x n)
        t_bottom, t_left, t_right, t_top (float):
            Temperaturas fijas en cada borde
        """
        N = n * n
        A = np.zeros((N, N), dtype=float)
        b = np.zeros(N, dtype=float)

        def k(i: int, j: int) -> int:
            """Indice lineal (0-based) para columna i y fila j (fila 0 = inferior)."""
            return j * n + i

        for j in range(n):
            for i in range(n):
                idx = k(i, j)
                A[idx, idx] = 4.0

                # vecino izquierdo
                if i - 1 >= 0:
                    A[idx, k(i - 1, j)] = -1.0
                else:
                    b[idx] += T_left

                # vecino derecho
                if i + 1 < n:
                    A[idx, k(i + 1, j)] = -1.0
                else:
                    b[idx] += T_right

                # vecino inferior
                if j - 1 >= 0:
                    A[idx, k(i, j - 1)] = -1.0
                else:
                    b[idx] += T_bottom

                # vecino superior
                if j + 1 < n:
                    A[idx, k(i, j + 1)] = -1.0
                else:
                    b[idx] += T_top

        return A, b

    @classmethod
    def solve_exact(
        cls, A: np.ndarray, b: np.ndarray, n: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def spectral_radius(
        cls, B: np.ndarray, tol: float = 1e-8, maxit: int = 10000
    ) -> float:
        """
        Calcula el radio espoectal de una matriz (el mayor valor absoluto de sus autovalores) usando el método de potencias iterativo.

        Parámetros:
        - B: matrtiz de ieteración.

        Retorns:
        - ρ (float): radio espectral de la matriz
        """
        n = B.shape[0]
        x = np.ones(n)

        for _ in range(maxit):
            x_new = np.dot(B, x)
            norm = np.max(np.abs(x_new))
            x_new = x_new / norm

            if np.max(np.abs(x_new - x)) < tol:
                return norm
            x = x_new
        return norm

    @classmethod
    def jacobi(
        cls,
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray | None = None,
        tol: float = 1e-6,
        maxit=10000,
    ) -> tuple[np.ndarray, int]:
        """
        Implementa el método iterativo de Jacobi para resolver Ax = b.

        Parámetros:
        - A: matriz de coeficientes.
        - b: vector del lado derecho.
        - x0: vector inicial (opcional, por defecto ceros).
        - tol: tolerancia para el criterio de convergencia.
        - maxit: número máximo de iteraciones.


        Retorna:
        - x: vector solución aproximada.
        - k: número de iteraciones realizadas.
        """
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)
        x = np.copy(x0)

        for k in range(maxit):
            x_new = np.zeros_like(x)
            for i in range(n):
                s = 0.0
                for j in range(n):
                    if j != i:
                        s += A[i, j] * x[j]
                x_new[i] = (b[i] - s) / A[i, i]

            if np.max(np.abs(x_new - x)) < tol:
                return x_new, k + 1
            x = x_new

        return x, maxit

    @staticmethod
    def spectral_radius_jacobi(A: np.ndarray) -> float:
        """
        Calcula el radio espectral del método de Jacobi.

        ρ_J = max(|λ_i(D⁻¹(L + U))|)

        Parámetros
        ----------
        A : np.ndarray
            Matriz del sistema.

        Retorna
        -------
        float
            Radio espectral del método de Jacobi.
        """
        D = np.diag(np.diag(A))
        L_U = A - D
        B = np.linalg.inv(D) @ L_U
        eigvals = np.linalg.eigvals(B)
        return max(abs(eigvals))

    @staticmethod
    def spectral_radius_gauss_seidel(A: np.ndarray) -> float:
        """
        Calcula el radio espectral del método de Gauss-Seidel.

        ρ_GS = max(|λ_i((D + L)⁻¹ U)|)

        Parámetros
        ----------
        A : np.ndarray
            Matriz del sistema.

        Retorna
        -------
        float
            Radio espectral del método de Gauss-Seidel.
        """
        D = np.diag(np.diag(A))
        L = np.tril(A, k=-1)
        U = np.triu(A, k=1)
        B = np.linalg.inv(D + L) @ U
        eigvals = np.linalg.eigvals(B)
        return max(abs(eigvals))


    @classmethod
    def gauss_seidel(
        cls,
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray | None = None,
        tol: float = 1e-8,
        maxit: int = 10000,
    ) -> tuple[np.ndarray, int]:
        """
        Implementa el método iterativo de Gauss-Seidel para resolver Ax = b.


        Parámetros:
        - A: matriz de coeficientes.
        - b: vector del lado derecho.
        - x0: vector inicial (opcional, por defecto ceros).
        - tol: tolerancia para el criterio de convergencia.
        - maxit: número máximo de iteraciones.


        Retorna:
        - x: vector solución aproximada.
        - k: número de iteraciones realizadas.
        """
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)
        x = np.copy(x0)

        for k in range(maxit):
            x_old = np.copy(x)
            for i in range(n):
                s1 = sum(A[i, j] * x[j] for j in range(i))
                s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
                x[i] = (b[i] - s1 - s2) / A[i, i]

            if np.max(np.abs(x - x_old)) < tol:
                return x, k + 1

        return x, maxit

    @classmethod
    def plot_temperature_distribution(cls, sol_exact: np.ndarray) -> None:
        """
        Genera una visualización gráfica de la distribución estacionaria de temperaturas.


        Parámetros:
        - sol_exact: matriz (n × n) con los valores de temperatura.
        """
        plt.imshow(sol_exact, origin="lower", cmap="coolwarm", interpolation="nearest")
        plt.colorbar(label="Temperatura")
        plt.title("Distribución estacionaria de temperaturas")
        plt.show()





def find_equilibria(A, beta=0.1, alpha=0.5, gamma=0.0111, psi=0.009, g = None):
    """
    Encuentra todos los puntos de equilibrio para un valor dado de A.
    
    Parámetros:
    - A: factor de crecimiento ambiental
    
    Retorna:
    - Lista de puntos de equilibrio (y donde g(y) = 0)
    """
    equilibria = []
    
    # y = 0 siempre es un equilibrio trivial
    if abs(g(0, A, beta, alpha, gamma, psi)) < 1e-10:
        equilibria.append(0.0)
    
    # Buscar equilibrios no triviales en diferentes intervalos
    # Probamos en rangos: [0.1, 20], [20, 50], [50, 100], [100, 200]
    search_ranges = [(0.1, 20), (20, 50), (50, 100), (100, 200)]
    
    for y_min, y_max in search_ranges:
        try:
            # Verificar si hay cambio de signo en el intervalo
            if g(y_min, A, beta, alpha, gamma, psi) * g(y_max, A, beta, alpha, gamma, psi) < 0:
                root = brentq(lambda y: g(y, A, beta, alpha, gamma, psi), y_min, y_max)
                # Evitar duplicados
                if not any(abs(root - eq) < 0.01 for eq in equilibria):
                    equilibria.append(root)
        except:
            pass
    
    # También usar fsolve con múltiples puntos iniciales
    initial_guesses = [1, 5, 10, 15, 20, 30, 50, 80, 100]
    for y0 in initial_guesses:
        try:
            root = fsolve(lambda y: g(y, A, beta, alpha, gamma, psi), y0)[0]
            if root > 0 and abs(g(root, A, beta, alpha, gamma, psi)) < 1e-8:
                # Verificar que no sea duplicado
                if not any(abs(root - eq) < 0.01 for eq in equilibria):
                    equilibria.append(root)
        except:
            pass
    
    return sorted(equilibria)


def runge_kutta_4(f, y0, t0, tf, h, A):
    """
    Método de Runge-Kutta de orden 4 para resolver dy/dt = f(y, A).
    
    Parámetros:
    - f: función que define dy/dt
    - y0: condición inicial
    - t0: tiempo inicial
    - tf: tiempo final
    - h: tamaño de paso
    - A: parámetro A del modelo
    
    Retorna:
    - t: array de tiempos
    - y: array de valores de y(t)
    """
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = h * f(y[i], A)
        k2 = h * f(y[i] + k1/2, A)
        k3 = h * f(y[i] + k2/2, A)
        k4 = h * f(y[i] + k3, A)
        
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y



def least_squares(x, y):
    """
    Calcula los coeficientes del modelo lineal y = a*x + b por mínimos cuadrados.
    Retorna: (a, b)
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x y y deben tener la misma longitud.")

    # Cálculo de coeficientes
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    b = y_mean - a * x_mean
    return a, b


def r_squared(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R^2.
    """
    ss_res = np.sum((y_real - y_pred)**2)
    ss_tot = np.sum((y_real - np.mean(y_real))**2)
    return 1 - (ss_res / ss_tot)


def best_model(x, y):
    """
    Evalúa distintas transformaciones de x y y y determina cuál modelo tiene
    el mejor ajuste lineal (mayor R^2).
    """
    if len(x) != len(y):
        raise ValueError("x y y deben tener la misma longitud.")
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    models = {
        "x vs y": (x, y),
        "x vs sqrt(y)": (x, np.sqrt(y)),
        "sqrt(x) vs y": (np.sqrt(x), y),
        "sqrt(x) vs sqrt(y)": (np.sqrt(x), np.sqrt(y)),
        "log(x) vs y": (np.log(x), y),
        "x vs log(y)": (x, np.log(y)),
        "log(x) vs log(y)": (np.log(x), np.log(y)),
        "x vs 1/y": (x, 1/y),
        "1/x vs y": (1/x, y),
        "1/x vs 1/y": (1/x, 1/y)
    }

    results = []

    for name, (X, Y) in models.items():
        # Validar que los valores sean finitos
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            continue
        
        try:
            a, b = least_squares(X, Y)
            y_pred = a * X + b
            r2 = r_squared(Y, y_pred)
            results.append((name, r2, a, b))
        except Exception:
            # Saltar combinaciones que fallen (log de negativos, división por cero, etc.)
            continue
    
    # Ordenar por mejor R²
    results.sort(key=lambda x: x[1], reverse=True)
    best = results[0]

    print("Resultados de los modelos evaluados:")
    for name, r2, a, b in results:
        print(f"{name:<20} → R² = {r2:.5f}")

    print("\n🏆 Mejor modelo encontrado:")
    print(f"Modelo: {best[0]}")
    print(f"R² = {best[1]:.5f}")
    print(f"Ecuación: y = {best[2]:.5f}x + {best[3]:.5f}")

    return best


def euler_sistema(f, y0: float, v0: float, t0: float, tf: float, h: float):
    """
    Método de Euler explícito para resolver un sistema de 2 EDOs de la forma:
    
        y' = f1(t, y, v)
        v' = f2(t, y, v)

    Parámetros
    ----------
    f : function
        Función que define el sistema: f(t, y, v) -> (dy/dt, dv/dt)
    y0 : float
        Condición inicial para la posición y(0)
    v0 : float
        Condición inicial para la velocidad v(0)
    t0, tf : float
        Tiempo inicial y final
    h : float
        Paso de integración

    Retorna
    -------
    t : ndarray
        Vector de tiempos
    y : ndarray
        Solución aproximada para la posición
    v : ndarray
        Solución aproximada para la velocidad
    """
    
    n_steps = int((tf - t0) / h) + 1

    t = np.linspace(t0, tf, n_steps)
    y = np.zeros(n_steps)
    v = np.zeros(n_steps)

    y[0] = y0
    v[0] = v0

    for i in range(n_steps - 1):
        dy_dt, dv_dt = f(t[i], y[i], v[i])
        
        y[i + 1] = y[i] + h * dy_dt
        v[i + 1] = v[i] + h * dv_dt
    
    return t, y, v

