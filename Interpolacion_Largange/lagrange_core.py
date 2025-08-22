import numpy as np
from typing import List, Tuple, Callable, Optional

def lagrange_basis(j: int, x: float, x_points: List[float]) -> float:
    """
    Calcula la j-ésima base de Lagrange para un punto x dado.
    
    Args:
        j: Índice de la base a calcular
        x: Punto donde evaluar la base
        x_points: Lista de puntos x de interpolación
    
    Returns:
        Valor de la j-ésima base de Lagrange en x
    """
    n = len(x_points)
    result = 1.0
    
    for i in range(n):
        if i != j:
            result *= (x - x_points[i]) / (x_points[j] - x_points[i])
    
    return result

def lagrange_interpolation(x: float, x_points: List[float], y_points: List[float]) -> float:
    """
    Calcula el valor del polinomio de interpolación de Lagrange en un punto x.
    
    Args:
        x: Punto donde evaluar el polinomio
        x_points: Lista de puntos x de interpolación
        y_points: Lista de valores y correspondientes
    
    Returns:
        Valor del polinomio de interpolación en x
    """
    n = len(x_points)
    result = 0.0
    
    for j in range(n):
        result += y_points[j] * lagrange_basis(j, x, x_points)
    
    return result

def get_lagrange_polynomial(x_points: List[float], y_points: List[float]) -> Callable[[float], float]:
    """
    Retorna una función que representa el polinomio de interpolación.
    
    Args:
        x_points: Lista de puntos x de interpolación
        y_points: Lista de valores y correspondientes
    
    Returns:
        Función que evalúa el polinomio de Lagrange en cualquier punto
    """
    def polynomial(x: float) -> float:
        return lagrange_interpolation(x, x_points, y_points)
    
    return polynomial

def get_all_basis_values(x_points: List[float], eval_points: List[float]) -> List[List[float]]:
    """
    Calcula todas las bases de Lagrange para un conjunto de puntos.
    
    Args:
        x_points: Puntos de interpolación
        eval_points: Puntos donde evaluar las bases
    
    Returns:
        Lista de listas con los valores de cada base en cada punto de evaluación
    """
    n = len(x_points)
    basis_values = []
    
    for j in range(n):
        basis_j = [lagrange_basis(j, x, x_points) for x in eval_points]
        basis_values.append(basis_j)
    
    return basis_values