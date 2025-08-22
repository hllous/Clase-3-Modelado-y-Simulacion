import numpy as np
from typing import Callable, List, Optional, Tuple

def forward_difference(f: Callable, x: float, h: float, order: int = 1) -> float:
    """
    Calcula la derivada utilizando diferencias finitas progresivas.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso
        order: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    Returns:
        Valor aproximado de la derivada
    """
    if order == 1:
        # Primera derivada con diferencias progresivas O(h)
        return (f(x + h) - f(x)) / h
    elif order == 2:
        # Segunda derivada con diferencias progresivas O(h)
        return (f(x + 2*h) - 2*f(x + h) + f(x)) / (h**2)
    elif order == 3:
        # Tercera derivada con diferencias progresivas O(h)
        return (f(x + 3*h) - 3*f(x + 2*h) + 3*f(x + h) - f(x)) / (h**3)
    elif order == 4:
        # Cuarta derivada con diferencias progresivas O(h)
        return (f(x + 4*h) - 4*f(x + 3*h) + 6*f(x + 2*h) - 4*f(x + h) + f(x)) / (h**4)
    else:
        raise ValueError(f"Orden {order} no soportado para diferencias progresivas")

def backward_difference(f: Callable, x: float, h: float, order: int = 1) -> float:
    """
    Calcula la derivada utilizando diferencias finitas regresivas.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso
        order: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    Returns:
        Valor aproximado de la derivada
    """
    if order == 1:
        # Primera derivada con diferencias regresivas O(h)
        return (f(x) - f(x - h)) / h
    elif order == 2:
        # Segunda derivada con diferencias regresivas O(h)
        return (f(x) - 2*f(x - h) + f(x - 2*h)) / (h**2)
    elif order == 3:
        # Tercera derivada con diferencias regresivas O(h)
        return (f(x) - 3*f(x - h) + 3*f(x - 2*h) - f(x - 3*h)) / (h**3)
    elif order == 4:
        # Cuarta derivada con diferencias regresivas O(h)
        return (f(x) - 4*f(x - h) + 6*f(x - 2*h) - 4*f(x - 3*h) + f(x - 4*h)) / (h**4)
    else:
        raise ValueError(f"Orden {order} no soportado para diferencias regresivas")

def central_difference(f: Callable, x: float, h: float, order: int = 1) -> float:
    """
    Calcula la derivada utilizando diferencias finitas centrales.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso
        order: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    Returns:
        Valor aproximado de la derivada
    """
    if order == 1:
        # Primera derivada con diferencias centrales O(h^2)
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        # Segunda derivada con diferencias centrales O(h^2)
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    elif order == 3:
        # Tercera derivada con diferencias centrales O(h^2)
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)
    elif order == 4:
        # Cuarta derivada con diferencias centrales O(h^2)
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)
    else:
        raise ValueError(f"Orden {order} no soportado para diferencias centrales")

def richardson_extrapolation(f: Callable, x: float, h: float, method: str, order: int = 1, levels: int = 2) -> float:
    """
    Aplica extrapolación de Richardson para mejorar la precisión de la aproximación numérica.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso inicial
        method: Método de diferencias ('forward', 'backward', 'central')
        order: Orden de la derivada
        levels: Niveles de extrapolación
    
    Returns:
        Valor mejorado de la derivada
    """
    # Determinar la función de diferencias a utilizar
    if method == 'forward':
        diff_func = forward_difference
        p = 1  # Orden de error para diferencias progresivas
    elif method == 'backward':
        diff_func = backward_difference
        p = 1  # Orden de error para diferencias regresivas
    elif method == 'central':
        diff_func = central_difference
        p = 2  # Orden de error para diferencias centrales
    else:
        raise ValueError(f"Método '{method}' no reconocido")
    
    # Inicializar la tabla de extrapolación
    D = np.zeros((levels, levels))
    
    # Calcular la primera columna con diferentes tamaños de paso
    for i in range(levels):
        h_i = h / (2**i)
        D[i, 0] = diff_func(f, x, h_i, order)
    
    # Aplicar extrapolación de Richardson
    for j in range(1, levels):
        for i in range(levels - j):
            # Fórmula de extrapolación de Richardson
            D[i, j] = D[i+1, j-1] + (D[i+1, j-1] - D[i, j-1]) / ((2**p) - 1)
    
    return D[0, levels-1]

def adaptive_step_size(f: Callable, x: float, method: str, order: int = 1, 
                      tol: float = 1e-6, max_iter: int = 10) -> Tuple[float, float]:
    """
    Determina un tamaño de paso adaptativo para obtener una derivada con precisión deseada.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        method: Método de diferencias ('forward', 'backward', 'central')
        order: Orden de la derivada
        tol: Tolerancia deseada
        max_iter: Número máximo de iteraciones
    
    Returns:
        Tupla (valor de la derivada, tamaño de paso utilizado)
    """
    # Determinar la función de diferencias a utilizar
    if method == 'forward':
        diff_func = forward_difference
    elif method == 'backward':
        diff_func = backward_difference
    elif method == 'central':
        diff_func = central_difference
    else:
        raise ValueError(f"Método '{method}' no reconocido")
    
    # Inicializar con un paso relativamente grande
    h = 0.1
    prev_deriv = diff_func(f, x, h, order)
    
    for i in range(max_iter):
        # Reducir el paso a la mitad
        h /= 2
        deriv = diff_func(f, x, h, order)
        
        # Verificar si la diferencia está dentro de la tolerancia
        if abs(deriv - prev_deriv) < tol:
            return deriv, h
            
        prev_deriv = deriv
    
    # Si no se alcanzó la tolerancia, devolver el mejor valor obtenido
    return deriv, h

def higher_order_forward(f: Callable, x: float, h: float, order: int = 1) -> float:
    """
    Calcula la derivada utilizando fórmulas de diferencias progresivas de mayor orden.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso
        order: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    Returns:
        Valor aproximado de la derivada con mayor precisión
    """
    if order == 1:
        # Primera derivada progresiva de orden O(h^2)
        return (-3*f(x) + 4*f(x + h) - f(x + 2*h)) / (2*h)
    elif order == 2:
        # Segunda derivada progresiva de orden O(h^2)
        return (2*f(x) - 5*f(x + h) + 4*f(x + 2*h) - f(x + 3*h)) / (h**2)
    else:
        raise ValueError(f"Orden {order} no soportado para diferencias progresivas de alto orden")

def higher_order_backward(f: Callable, x: float, h: float, order: int = 1) -> float:
    """
    Calcula la derivada utilizando fórmulas de diferencias regresivas de mayor orden.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso
        order: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    Returns:
        Valor aproximado de la derivada con mayor precisión
    """
    if order == 1:
        # Primera derivada regresiva de orden O(h^2)
        return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)
    elif order == 2:
        # Segunda derivada regresiva de orden O(h^2)
        return (2*f(x) - 5*f(x - h) + 4*f(x - 2*h) - f(x - 3*h)) / (h**2)
    else:
        raise ValueError(f"Orden {order} no soportado para diferencias regresivas de alto orden")

def higher_order_central(f: Callable, x: float, h: float, order: int = 1) -> float:
    """
    Calcula la derivada utilizando fórmulas de diferencias centrales de mayor orden.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        h: Tamaño del paso
        order: Orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    Returns:
        Valor aproximado de la derivada con mayor precisión
    """
    if order == 1:
        # Primera derivada central de orden O(h^4)
        return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)
    elif order == 2:
        # Segunda derivada central de orden O(h^4)
        return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12*h**2)
    elif order == 4:
        # Cuarta derivada central de orden O(h^4)
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)
    else:
        raise ValueError(f"Orden {order} no soportado para diferencias centrales de alto orden")