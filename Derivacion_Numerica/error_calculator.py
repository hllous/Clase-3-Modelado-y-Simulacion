import numpy as np
from typing import Callable, List, Tuple, Optional
import sympy as sp

def truncation_error(method: str, h: float, order: int = 1) -> float:
    """
    Calcula el error de truncamiento basado en el método y el tamaño del paso.
    
    Args:
        method: Método de diferencias ('forward', 'backward', 'central')
        h: Tamaño del paso
        order: Orden de la derivada
    
    Returns:
        Orden del error de truncamiento
    """
    if method in ['forward', 'backward']:
        # Diferencias progresivas y regresivas tienen error O(h)
        return h
    elif method == 'central':
        # Diferencias centrales tienen error O(h^2)
        return h**2
    elif method in ['higher_forward', 'higher_backward']:
        # Diferencias progresivas y regresivas de alto orden tienen error O(h^2)
        return h**2
    elif method == 'higher_central':
        # Diferencias centrales de alto orden tienen error O(h^4)
        return h**4
    else:
        raise ValueError(f"Método '{method}' no reconocido")

def exact_derivative(expr_str: str, var_name: str, x_val: float, order: int = 1) -> float:
    """
    Calcula la derivada exacta de una expresión simbólica.
    
    Args:
        expr_str: Expresión como cadena de texto
        var_name: Nombre de la variable
        x_val: Valor de x donde evaluar la derivada
        order: Orden de la derivada
    
    Returns:
        Valor exacto de la derivada
    """
    # Crear la variable simbólica
    var = sp.Symbol(var_name)
    
    # Convertir la cadena a expresión simbólica
    expr = sp.sympify(expr_str)
    
    # Calcular la derivada
    for _ in range(order):
        expr = sp.diff(expr, var)
    
    # Convertir a función numérica y evaluar
    f_derivative = sp.lambdify(var, expr, "numpy")
    return float(f_derivative(x_val))

def calculate_error(approx_value: float, exact_value: float) -> Tuple[float, float]:
    """
    Calcula el error absoluto y relativo.
    
    Args:
        approx_value: Valor aproximado
        exact_value: Valor exacto
    
    Returns:
        Tupla (error absoluto, error relativo)
    """
    abs_error = abs(approx_value - exact_value)
    
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = np.nan
    
    return abs_error, rel_error

def optimal_step_size(f_exact: Callable, x: float, method: str, order: int = 1, 
                     tol: float = 1e-6, start_h: float = 0.1) -> float:
    """
    Encuentra el tamaño de paso óptimo para minimizar el error total.
    
    Args:
        f_exact: Función de la derivada exacta
        x: Punto donde calcular la derivada
        method: Método de diferencias
        order: Orden de la derivada
        tol: Tolerancia deseada
        start_h: Tamaño de paso inicial
    
    Returns:
        Tamaño de paso óptimo
    """
    from derivative_core import forward_difference, backward_difference, central_difference
    
    # Determinar la función de diferencias a utilizar
    if method == 'forward':
        diff_func = forward_difference
    elif method == 'backward':
        diff_func = backward_difference
    elif method == 'central':
        diff_func = central_difference
    else:
        raise ValueError(f"Método '{method}' no reconocido")
    
    # Método simple de búsqueda
    h = start_h
    exact_deriv = f_exact(x)
    
    # Inicializar con un error grande
    best_h = h
    best_error = abs(diff_func(lambda x: f_exact(x), x, h, order) - exact_deriv)
    
    # Probar diferentes tamaños de paso
    for i in range(20):  # Intentar con 20 tamaños diferentes
        h /= 2
        approx_deriv = diff_func(lambda x: f_exact(x), x, h, order)
        error = abs(approx_deriv - exact_deriv)
        
        if error < best_error:
            best_error = error
            best_h = h
            
        # Si el error empieza a aumentar debido a errores de redondeo, terminar
        if i > 2 and error > 2 * best_error:
            break
            
    return best_h