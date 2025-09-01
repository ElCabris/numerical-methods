import math
import sympy as sp
from typing import Union, Tuple, Callable

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

def taylor_remainder_bound(f_expr: sp.Expr, var: sp.Symbol, a: Number, n: int, x: Number) -> Tuple[sp.Expr, float]:
    """
    f_expr: Symbolic function
    var: sp.Symbol
    a: Expansion point
    n: order of the polynomial
    x: point where the height is evaluated

    Return
    ---
    deriv_n1_expr: The (n+1)th symbolic derivative.
    bound: Numerical bound of |R_n(x)|.
    """

    if n < 0:
        raise ValueError("")

    a_num = float(sp.N(a))
    x_num = float(sp.N(x))
    lo, hi = (a_num, x_num) if a_num <= x_num else (x_num, a_num)
    if lo == hi:
        return sp.diff(f_expr, var, n+1), 0.0

    deriv_n1_expr = sp.simplify(sp.diff(f_expr, var, n+1))

    crit_points = []
    try:
        crit_set = sp.solveset(sp.Eq(sp.diff(deriv_n1_expr, var), 0), var, domain=sp.Interval(lo, hi))

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
    bound = M / math.factorial(n + 1) * abs(x_num - a_num) ** (n+1)
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
        return bisection(f,c, b, tol)
    

def bisection_iters(a: float, b: float, eps: float) -> int:
    if eps <= 0:
        raise ValueError("eps debe ser > 0")
    L = abs(b - a)
    if eps >= L:
        return 0
    return math.ceil(math.log(L/eps, 2))

def false_position(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    """
    """
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

    raise ValueError("No se encontró la raíz en {} iteraciones".format(max_iter))

def secante(f, x0, x1, tol=1e-10, max_iter=100):
    """
    Encuentra una raíz de f(x)=0 usando el método de la secante
    :param f: función a resolver
    :param x0: primera aproximación inicial
    :param x1: segunda aproximación inicial
    :param tol: tolerancia
    :param max_iter: máximo de iteraciones
    :return: aproximación de la raíz, número de iteraciones
    """
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            raise ValueError("División por cero en iteración {}".format(i))

        # Fórmula de la secante
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        # Revisar convergencia
        if abs(x2 - x1) < tol:
            return x2, i+1

        # Avanzar
        x0, x1 = x1, x2

    raise RuntimeError("El método no convergió en {} iteraciones".format(max_iter))