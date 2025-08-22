import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional, Dict
from derivative_core import forward_difference, backward_difference, central_difference

def plot_derivatives(f: Callable, x_range: Tuple[float, float], 
                     methods: List[str], h: float, order: int = 1,
                     f_exact: Optional[Callable] = None) -> plt.Figure:
    """
    Genera un gráfico comparando diferentes métodos de derivación.
    
    Args:
        f: Función a derivar
        x_range: Rango de x para graficar (inicio, fin)
        methods: Lista de métodos a comparar ('forward', 'backward', 'central')
        h: Tamaño del paso
        order: Orden de la derivada
        f_exact: Función de la derivada exacta (opcional)
    
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Crear puntos para graficar
    x_points = np.linspace(x_range[0], x_range[1], 500)
    
    # Métodos disponibles
    method_funcs = {
        'forward': forward_difference,
        'backward': backward_difference,
        'central': central_difference
    }
    
    method_colors = {
        'forward': 'b',
        'backward': 'r',
        'central': 'g',
        'exact': 'k'
    }
    
    method_labels = {
        'forward': 'Diferencias progresivas',
        'backward': 'Diferencias regresivas',
        'central': 'Diferencias centrales',
        'exact': 'Derivada exacta'
    }
    
    # Graficar cada método
    for method in methods:
        if method in method_funcs:
            y_values = []
            for x in x_points:
                try:
                    y = method_funcs[method](f, x, h, order)
                    y_values.append(y)
                except:
                    y_values.append(np.nan)
            
            ax.plot(x_points, y_values, method_colors[method], 
                   label=f"{method_labels[method]} (h={h})")
    
    # Graficar la derivada exacta si está disponible
    if f_exact:
        y_exact = [f_exact(x) for x in x_points]
        ax.plot(x_points, y_exact, method_colors['exact'], linestyle='--', 
               label=method_labels['exact'])
    
    # Configurar gráfico
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel(f'f{"".join(["'" for _ in range(order)])}(x)')
    ax.set_title(f'Comparación de métodos de derivación (orden {order})')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_error_vs_h(f: Callable, x: float, methods: List[str], 
                   h_range: Tuple[float, float], order: int = 1,
                   f_exact: Optional[Callable] = None) -> plt.Figure:
    """
    Genera un gráfico del error en función del tamaño del paso.
    
    Args:
        f: Función a derivar
        x: Punto donde calcular la derivada
        methods: Lista de métodos a comparar
        h_range: Rango de h para graficar (mínimo, máximo)
        order: Orden de la derivada
        f_exact: Función de la derivada exacta
    
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Métodos disponibles
    method_funcs = {
        'forward': forward_difference,
        'backward': backward_difference,
        'central': central_difference
    }
    
    method_colors = {
        'forward': 'b',
        'backward': 'r',
        'central': 'g'
    }
    
    method_labels = {
        'forward': 'Diferencias progresivas',
        'backward': 'Diferencias regresivas',
        'central': 'Diferencias centrales'
    }
    
    # Verificar que la derivada exacta esté disponible
    if f_exact is None:
        raise ValueError("Se requiere la derivada exacta para calcular el error")
    
    exact_value = f_exact(x)
    
    # Crear escala logarítmica para h
    h_values = np.logspace(np.log10(h_range[0]), np.log10(h_range[1]), 100)
    
    # Graficar el error para cada método
    for method in methods:
        if method in method_funcs:
            errors = []
            for h in h_values:
                try:
                    approx = method_funcs[method](f, x, h, order)
                    error = abs(approx - exact_value)
                    errors.append(error)
                except:
                    errors.append(np.nan)
            
            ax.loglog(h_values, errors, method_colors[method], 
                     label=method_labels[method])
    
    # Graficar líneas de referencia para O(h) y O(h^2)
    h_ref = np.logspace(np.log10(h_range[0]), np.log10(h_range[1]), 2)
    c = 0.1  # Constante de escala
    
    ax.loglog(h_ref, c * h_ref, 'k--', label='O(h)')
    ax.loglog(h_ref, c * h_ref**2, 'k:', label='O(h²)')
    
    # Configurar gráfico
    ax.legend(loc='best')
    ax.set_xlabel('h (tamaño del paso)')
    ax.set_ylabel('Error absoluto')
    ax.set_title(f'Error vs tamaño del paso en x={x}')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_function_and_derivative(f: Callable, x_range: Tuple[float, float], 
                               method: str, h: float, order: int = 1,
                               f_exact: Optional[Callable] = None) -> plt.Figure:
    """
    Genera un gráfico que muestra la función original y su derivada.
    
    Args:
        f: Función original
        x_range: Rango de x para graficar (inicio, fin)
        method: Método de derivación
        h: Tamaño del paso
        order: Orden de la derivada
        f_exact: Función de la derivada exacta (opcional)
    
    Returns:
        Figura de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Crear puntos para graficar
    x_points = np.linspace(x_range[0], x_range[1], 500)
    
    # Graficar la función original
    y_values = [f(x) for x in x_points]
    ax1.plot(x_points, y_values, 'b-', label='Función f(x)')
    
    # Métodos disponibles
    method_funcs = {
        'forward': forward_difference,
        'backward': backward_difference,
        'central': central_difference
    }
    
    # Graficar la derivada calculada
    if method in method_funcs:
        y_deriv = []
        for x in x_points:
            try:
                y = method_funcs[method](f, x, h, order)
                y_deriv.append(y)
            except:
                y_deriv.append(np.nan)
        
        ax2.plot(x_points, y_deriv, 'r-', 
               label=f'Derivada calculada (método: {method}, h={h})')
    
    # Graficar la derivada exacta si está disponible
    if f_exact:
        y_exact = [f_exact(x) for x in x_points]
        ax2.plot(x_points, y_exact, 'g--', label='Derivada exacta')
    
    # Configurar gráficos
    ax1.legend(loc='best')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Función original')
    ax1.grid(True, alpha=0.3)
    
    ax2.legend(loc='best')
    ax2.set_xlabel('x')
    ax2.set_ylabel(f'f{"".join(["'" for _ in range(order)])}(x)')
    ax2.set_title(f'Derivada de orden {order}')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig