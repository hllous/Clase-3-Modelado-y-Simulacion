import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from lagrange_core import lagrange_interpolation, get_all_basis_values

def plot_interpolation(x_points: List[float], y_points: List[float], 
                      original_func: Optional[Callable] = None,
                      title: str = "Interpolación de Lagrange",
                      show_basis: bool = False) -> plt.Figure:
    """
    Genera un gráfico con el polinomio de interpolación y opcionalmente la función original.
    
    Args:
        x_points: Puntos x de interpolación
        y_points: Valores y correspondientes
        original_func: Función original (opcional)
        title: Título del gráfico
        show_basis: Mostrar las bases de Lagrange
    
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Configurar límites de la gráfica
    x_min, x_max = min(x_points) - 0.5, max(x_points) + 0.5
    eval_points = np.linspace(x_min, x_max, 1000)
    
    # Graficar los puntos de interpolación
    ax.scatter(x_points, y_points, color='red', zorder=5, s=50, label='Puntos de interpolación')
    
    # Graficar el polinomio de interpolación
    interp_values = [lagrange_interpolation(x, x_points, y_points) for x in eval_points]
    ax.plot(eval_points, interp_values, 'b-', linewidth=2, label='Polinomio de Lagrange')
    
    # Graficar la función original si está disponible
    if original_func:
        # Manejo seguro de evaluación de funciones
        original_values = []
        valid_points = []
        
        for x in eval_points:
            try:
                value = original_func(x)
                # Verificar si el valor es válido (no NaN, no infinito)
                if np.isfinite(value):
                    original_values.append(value)
                    valid_points.append(x)
            except:
                # Ignorar puntos donde la función no se puede evaluar
                continue
        
        if valid_points:
            ax.plot(valid_points, original_values, 'g--', linewidth=1.5, label='Función original')
    
    # Graficar bases de Lagrange si se solicita
    if show_basis:
        basis_values = get_all_basis_values(x_points, eval_points)
        for i, basis in enumerate(basis_values):
            ax.plot(eval_points, basis, '--', linewidth=0.7, alpha=0.6, 
                   label=f'Base L_{i}(x)')
    
    # Configurar leyenda y etiquetas
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Añadir líneas de ayuda
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    
    return fig

def plot_error(x_points: List[float], y_points: List[float], 
              original_func: Callable, 
              title: str = "Error de Interpolación") -> plt.Figure:
    """
    Genera un gráfico del error de interpolación.
    
    Args:
        x_points: Puntos x de interpolación
        y_points: Valores y correspondientes
        original_func: Función original
        title: Título del gráfico
    
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Configurar límites de la gráfica
    x_min, x_max = min(x_points) - 0.5, max(x_points) + 0.5
    eval_points = np.linspace(x_min, x_max, 1000)
    
    # Calcular errores con manejo seguro de funciones
    valid_points = []
    errors = []
    
    for x in eval_points:
        try:
            interp_value = lagrange_interpolation(x, x_points, y_points)
            orig_value = original_func(x)
            
            # Verificar valores válidos
            if np.isfinite(interp_value) and np.isfinite(orig_value):
                valid_points.append(x)
                errors.append(abs(interp_value - orig_value))
        except:
            continue
    
    # Graficar el error
    if valid_points:
        ax.plot(valid_points, errors, 'r-', linewidth=1.5, label='Error absoluto')
    
    # Marcar los puntos de interpolación
    ax.scatter(x_points, [0] * len(x_points), color='blue', zorder=5, s=30, 
              label='Puntos de interpolación')
    
    # Configurar leyenda y etiquetas
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('Error absoluto')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig