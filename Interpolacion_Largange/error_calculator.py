import numpy as np
from typing import List, Tuple, Callable, Optional
from lagrange_core import lagrange_interpolation

def local_error(x: float, x_points: List[float], y_points: List[float], 
                original_func: Optional[Callable] = None) -> float:
    """
    Calcula el error local en un punto específico.
    Si se proporciona la función original, calcula la diferencia.
    Si no, devuelve 0 para los puntos de interpolación y None para otros puntos.
    
    Args:
        x: Punto donde calcular el error
        x_points: Puntos x de interpolación
        y_points: Valores y correspondientes
        original_func: Función original (opcional)
    
    Returns:
        Error local en el punto x
    """
    if original_func is None:
        # Sin función original, verificar si es un punto de interpolación
        if x in x_points:
            idx = x_points.index(x)
            interpolated = lagrange_interpolation(x, x_points, y_points)
            return abs(interpolated - y_points[idx])
        return None
    else:
        # Con función original, calcular la diferencia
        interpolated = lagrange_interpolation(x, x_points, y_points)
        original = original_func(x)
        return abs(interpolated - original)

def global_error(x_points: List[float], y_points: List[float], 
                original_func: Callable, eval_points: List[float]) -> Tuple[float, float, float]:
    """
    Calcula diferentes medidas de error global.
    
    Args:
        x_points: Puntos x de interpolación
        y_points: Valores y correspondientes
        original_func: Función original
        eval_points: Puntos donde evaluar el error
    
    Returns:
        Tupla con (error máximo, error promedio, error cuadrático medio)
    """
    errors = [local_error(x, x_points, y_points, original_func) for x in eval_points]
    
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    mse = sum(e**2 for e in errors) / len(errors)
    
    return max_error, avg_error, mse

def error_bound(x: float, x_points: List[float], derivative_func: Callable, 
               order: Optional[int] = None) -> float:
    """
    Calcula la cota del error de interpolación usando el teorema del error.
    
    Args:
        x: Punto donde calcular la cota
        x_points: Puntos x de interpolación
        derivative_func: Función derivada de orden n+1
        order: Orden de la derivada (por defecto es n+1)
    
    Returns:
        Cota del error en el punto x
    """
    n = len(x_points)
    if order is None:
        order = n + 1  # Corregido: usar derivada de orden n+1
    
    # Calcular el término (x - x_0)(x - x_1)...(x - x_{n-1})
    omega = 1.0
    for x_i in x_points:
        omega *= (x - x_i)
    
    # Buscar M que sea el máximo de la derivada n+1 en el intervalo
    a, b = min(min(x_points), x), max(max(x_points), x)
    sample_points = np.linspace(a, b, 1000)
    
    # Protección contra errores en la evaluación de la derivada
    max_derivative = 0
    for t in sample_points:
        try:
            deriv_value = abs(derivative_func(t))
            if np.isfinite(deriv_value) and deriv_value > max_derivative:
                max_derivative = deriv_value
        except:
            continue
    
    if max_derivative == 0:
        return 0.0  # No se pudo calcular la derivada máxima
    
    # Calcular la cota del error
    try:
        factorial_term = np.math.factorial(n)
        bound = (max_derivative / factorial_term) * abs(omega)
        return bound
    except OverflowError:
        # Manejar el caso de factoriales grandes
        log_factorial = sum(np.log(i) for i in range(1, n+1))
        log_bound = np.log(max_derivative) - log_factorial + np.log(abs(omega))
        return np.exp(log_bound)