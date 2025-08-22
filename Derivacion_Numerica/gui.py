import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Callable, List, Tuple, Optional, Dict
import sympy as sp
import re
import json

from derivative_core import (
    forward_difference, backward_difference, central_difference,
    richardson_extrapolation, adaptive_step_size,
    higher_order_forward, higher_order_backward, higher_order_central
)
from error_calculator import truncation_error, exact_derivative, calculate_error, optimal_step_size
from plotter import plot_derivatives, plot_error_vs_h, plot_function_and_derivative

class DerivativeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Derivación Numérica")
        self.geometry("1200x800")
        self.configure(padx=10, pady=10)
        
        self.function = None
        self.exact_derivative_func = None
        
        self.create_variables()
        self.create_widgets()
        self.setup_layout()
    
    def create_variables(self):
        """Inicializa variables del programa"""
        self.func_str = tk.StringVar(value="")
        self.x_value = tk.StringVar(value="0")
        self.h_value = tk.StringVar(value="0.1")
        self.order_value = tk.IntVar(value=1)
        
        self.use_forward = tk.BooleanVar(value=True)
        self.use_backward = tk.BooleanVar(value=True)
        self.use_central = tk.BooleanVar(value=True)
        self.use_higher_order = tk.BooleanVar(value=False)
        self.use_richardson = tk.BooleanVar(value=False)
        
        self.plot_type = tk.StringVar(value="comparison")
        self.test_function_var = tk.StringVar(value="sin")
    
    def create_widgets(self):
        """Crea todos los widgets de la interfaz"""
        # Frame principal dividido en dos partes
        self.main_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        
        # Panel izquierdo para controles
        self.control_frame = ttk.Frame(self.main_frame, padding=5)
        
        # Panel derecho para gráficos
        self.graph_frame = ttk.Frame(self.main_frame, padding=5)
        
        # Agregar frames al panel principal
        self.main_frame.add(self.control_frame, weight=1)
        self.main_frame.add(self.graph_frame, weight=2)
        
        # --- Controles de entrada ---
        # Frame para función
        self.func_frame = ttk.LabelFrame(self.control_frame, text="Función a derivar", padding=5)
        
        ttk.Label(self.func_frame, text="f(x) = ").grid(row=0, column=0, padx=5, pady=5)
        self.func_entry = ttk.Entry(self.func_frame, textvariable=self.func_str, width=30)
        self.func_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.set_func_btn = ttk.Button(self.func_frame, text="Establecer función", 
                                      command=self.set_function)
        self.set_func_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Label(self.func_frame, text="Ejemplos: sin(x), x**2, exp(-x), x**3 - 2*x + 1").grid(
            row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Frame para datos de prueba
        self.test_frame = ttk.LabelFrame(self.control_frame, text="Datos de prueba", padding=5)
        
        ttk.Label(self.test_frame, text="Función:").grid(row=0, column=0, padx=5, pady=5)
        self.test_function_combo = ttk.Combobox(self.test_frame, textvariable=self.test_function_var,
                                               values=["sin", "exp", "poly", "sqrt", "log"])
        self.test_function_combo.grid(row=0, column=1, padx=5, pady=5)
        
        self.generate_test_btn = ttk.Button(self.test_frame, text="Usar función de prueba",
                                           command=self.generate_test_function)
        self.generate_test_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Frame para parámetros de derivación
        self.params_frame = ttk.LabelFrame(self.control_frame, text="Parámetros de derivación", padding=5)
        
        ttk.Label(self.params_frame, text="Valor de x:").grid(row=0, column=0, padx=5, pady=5)
        self.x_entry = ttk.Entry(self.params_frame, textvariable=self.x_value, width=10)
        self.x_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.params_frame, text="Tamaño de paso h:").grid(row=1, column=0, padx=5, pady=5)
        self.h_entry = ttk.Entry(self.params_frame, textvariable=self.h_value, width=10)
        self.h_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.params_frame, text="Orden de la derivada:").grid(row=2, column=0, padx=5, pady=5)
        self.order_spinbox = ttk.Spinbox(self.params_frame, from_=1, to=4, 
                                         textvariable=self.order_value, width=5)
        self.order_spinbox.grid(row=2, column=1, padx=5, pady=5)
        
        # Frame para métodos de derivación
        self.methods_frame = ttk.LabelFrame(self.control_frame, text="Métodos de derivación", padding=5)
        
        ttk.Checkbutton(self.methods_frame, text="Diferencias progresivas", 
                       variable=self.use_forward).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(self.methods_frame, text="Diferencias regresivas", 
                       variable=self.use_backward).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(self.methods_frame, text="Diferencias centrales", 
                       variable=self.use_central).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(self.methods_frame, text="Fórmulas de mayor orden", 
                       variable=self.use_higher_order).grid(row=3, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(self.methods_frame, text="Extrapolación de Richardson", 
                       variable=self.use_richardson).grid(row=4, column=0, sticky="w", padx=5, pady=2)
        
        # Frame para opciones de visualización
        self.viz_frame = ttk.LabelFrame(self.control_frame, text="Opciones de visualización", padding=5)
        
        ttk.Radiobutton(self.viz_frame, text="Comparar métodos", 
                       variable=self.plot_type, value="comparison").grid(
            row=0, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Radiobutton(self.viz_frame, text="Error vs tamaño de paso", 
                       variable=self.plot_type, value="error_h").grid(
            row=1, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Radiobutton(self.viz_frame, text="Función y derivada", 
                       variable=self.plot_type, value="function_deriv").grid(
            row=2, column=0, sticky="w", padx=5, pady=2)
        
        # Botones de acción
        self.action_frame = ttk.Frame(self.control_frame, padding=5)
        
        self.calculate_btn = ttk.Button(self.action_frame, text="Calcular derivadas", 
                                       command=self.calculate_derivatives)
        self.calculate_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.optimal_h_btn = ttk.Button(self.action_frame, text="Paso óptimo", 
                                      command=self.find_optimal_h)
        self.optimal_h_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.exit_btn = ttk.Button(self.action_frame, text="Salir", 
                                  command=self.quit)
        self.exit_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Frame para resultados
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Resultados", padding=5)
        
        self.results_text = tk.Text(self.results_frame, height=15, width=40, wrap=tk.WORD)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, 
                                              command=self.results_text.yview)
        self.results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=self.results_scrollbar.set)
        
        # --- Panel de gráficos ---
        self.canvas_frame = ttk.Frame(self.graph_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas para los gráficos
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Esperando cálculos...")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar para manipular el gráfico
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
    
    def setup_layout(self):
        """Configura el layout general de la aplicación"""
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar layout del panel de control
        self.func_frame.pack(fill=tk.X, pady=5)
        self.test_frame.pack(fill=tk.X, pady=5)
        self.params_frame.pack(fill=tk.X, pady=5)
        self.methods_frame.pack(fill=tk.X, pady=5)
        self.viz_frame.pack(fill=tk.X, pady=5)
        self.action_frame.pack(fill=tk.X, pady=5)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def set_function(self):
        """Establece la función a derivar y su derivada exacta"""
        try:
            func_str = self.func_str.get().strip()
            
            if not func_str:
                messagebox.showerror("Error", "Ingrese una función válida")
                return
            
            # Crear función con sympy y convertir a función numérica
            x = sp.Symbol('x')
            
            # Limpiar las expresiones de funciones comunes
            func_str = func_str.replace("^", "**")
            
            # Crear expresiones simbólicas
            try:
                func_expr = sp.sympify(func_str)
            except:
                messagebox.showerror("Error", "No se pudo interpretar la función. Verifique la sintaxis.")
                return
            
            # Crear la función de derivada exacta según el orden
            order = self.order_value.get()
            deriv_expr = func_expr
            for _ in range(order):
                deriv_expr = sp.diff(deriv_expr, x)
            
            # Convertir a funciones numéricas con manejo de excepciones
            def safe_func(x_val):
                try:
                    result = float(func_expr.subs(x, x_val))
                    return result if np.isfinite(result) else np.nan
                except:
                    return np.nan
            
            def safe_deriv(x_val):
                try:
                    result = float(deriv_expr.subs(x, x_val))
                    return result if np.isfinite(result) else np.nan
                except:
                    return np.nan
            
            # Asignar las funciones
            self.function = safe_func
            self.exact_derivative_func = safe_deriv
            
            # Mostrar información en el área de resultados
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Función establecida: f(x) = {func_str}\n\n")
            self.results_text.insert(tk.END, f"Derivada exacta de orden {order}:\n")
            self.results_text.insert(tk.END, f"f{''.join(['\''] * order)}(x) = {deriv_expr}\n\n")
            
            messagebox.showinfo("Información", "Función establecida correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al establecer la función: {str(e)}")
            self.function = None
            self.exact_derivative_func = None
    
    def generate_test_function(self):
        """Genera una función de prueba predefinida"""
        function_type = self.test_function_var.get()
        
        if function_type == "sin":
            self.func_str.set("sin(x)")
        elif function_type == "exp":
            self.func_str.set("exp(x)")
        elif function_type == "poly":
            self.func_str.set("x**3 - 2*x + 1")
        elif function_type == "sqrt":
            self.func_str.set("sqrt(x)")
        elif function_type == "log":
            self.func_str.set("log(x)")
        
        # Establecer la función
        self.set_function()
    
    def calculate_derivatives(self):
        """Calcula las derivadas según los métodos seleccionados y muestra resultados"""
        if not self.function:
            messagebox.showerror("Error", "Primero debe establecer una función válida")
            return
        
        try:
            # Obtener parámetros
            x = float(self.x_value.get())
            h = float(self.h_value.get())
            order = self.order_value.get()
            
            if h <= 0:
                messagebox.showerror("Error", "El tamaño de paso h debe ser positivo")
                return
            
            # Limpiar resultados anteriores
            self.results_text.delete(1.0, tk.END)
            
            # Calcular derivada exacta
            exact_value = None
            if self.exact_derivative_func:
                exact_value = self.exact_derivative_func(x)
                self.results_text.insert(tk.END, f"Valor exacto de la derivada en x={x}:\n")
                self.results_text.insert(tk.END, f"f{''.join(['\''] * order)}({x}) = {exact_value:.10e}\n\n")
            
            # Calcular derivadas numéricas
            self.results_text.insert(tk.END, f"Derivadas numéricas en x={x} con h={h}:\n\n")
            
            results = {}
            
            # Métodos básicos
            if self.use_forward.get():
                try:
                    forward_value = forward_difference(self.function, x, h, order)
                    results['forward'] = forward_value
                    self.results_text.insert(tk.END, f"Diferencias progresivas: {forward_value:.10e}\n")
                    
                    if exact_value is not None:
                        abs_error, rel_error = calculate_error(forward_value, exact_value)
                        self.results_text.insert(tk.END, f"  Error absoluto: {abs_error:.10e}\n")
                        self.results_text.insert(tk.END, f"  Error relativo: {rel_error:.10e}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error en diferencias progresivas: {str(e)}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            if self.use_backward.get():
                try:
                    backward_value = backward_difference(self.function, x, h, order)
                    results['backward'] = backward_value
                    self.results_text.insert(tk.END, f"Diferencias regresivas: {backward_value:.10e}\n")
                    
                    if exact_value is not None:
                        abs_error, rel_error = calculate_error(backward_value, exact_value)
                        self.results_text.insert(tk.END, f"  Error absoluto: {abs_error:.10e}\n")
                        self.results_text.insert(tk.END, f"  Error relativo: {rel_error:.10e}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error en diferencias regresivas: {str(e)}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            if self.use_central.get():
                try:
                    central_value = central_difference(self.function, x, h, order)
                    results['central'] = central_value
                    self.results_text.insert(tk.END, f"Diferencias centrales: {central_value:.10e}\n")
                    
                    if exact_value is not None:
                        abs_error, rel_error = calculate_error(central_value, exact_value)
                        self.results_text.insert(tk.END, f"  Error absoluto: {abs_error:.10e}\n")
                        self.results_text.insert(tk.END, f"  Error relativo: {rel_error:.10e}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error en diferencias centrales: {str(e)}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            # Métodos avanzados
            if self.use_higher_order.get():
                self.results_text.insert(tk.END, "Fórmulas de mayor orden:\n")
                
                try:
                    higher_forward = higher_order_forward(self.function, x, h, order)
                    results['higher_forward'] = higher_forward
                    self.results_text.insert(tk.END, f"  Progresivas O(h²): {higher_forward:.10e}\n")
                    
                    if exact_value is not None:
                        abs_error, rel_error = calculate_error(higher_forward, exact_value)
                        self.results_text.insert(tk.END, f"    Error absoluto: {abs_error:.10e}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"  Error en progresivas de alto orden: {str(e)}\n")
                
                try:
                    higher_backward = higher_order_backward(self.function, x, h, order)
                    results['higher_backward'] = higher_backward
                    self.results_text.insert(tk.END, f"  Regresivas O(h²): {higher_backward:.10e}\n")
                    
                    if exact_value is not None:
                        abs_error, rel_error = calculate_error(higher_backward, exact_value)
                        self.results_text.insert(tk.END, f"    Error absoluto: {abs_error:.10e}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"  Error en regresivas de alto orden: {str(e)}\n")
                
                try:
                    higher_central = higher_order_central(self.function, x, h, order)
                    results['higher_central'] = higher_central
                    self.results_text.insert(tk.END, f"  Centrales O(h⁴): {higher_central:.10e}\n")
                    
                    if exact_value is not None:
                        abs_error, rel_error = calculate_error(higher_central, exact_value)
                        self.results_text.insert(tk.END, f"    Error absoluto: {abs_error:.10e}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"  Error en centrales de alto orden: {str(e)}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            if self.use_richardson.get():
                self.results_text.insert(tk.END, "Extrapolación de Richardson:\n")
                
                # Aplicar extrapolación de Richardson a los métodos seleccionados
                if self.use_forward.get():
                    try:
                        rich_forward = richardson_extrapolation(self.function, x, h, 'forward', order)
                        results['richardson_forward'] = rich_forward
                        self.results_text.insert(tk.END, f"  Progresivas mejoradas: {rich_forward:.10e}\n")
                        
                        if exact_value is not None:
                            abs_error, rel_error = calculate_error(rich_forward, exact_value)
                            self.results_text.insert(tk.END, f"    Error absoluto: {abs_error:.10e}\n")
                    except Exception as e:
                        self.results_text.insert(tk.END, f"  Error en extrapolación de progresivas: {str(e)}\n")
                
                if self.use_backward.get():
                    try:
                        rich_backward = richardson_extrapolation(self.function, x, h, 'backward', order)
                        results['richardson_backward'] = rich_backward
                        self.results_text.insert(tk.END, f"  Regresivas mejoradas: {rich_backward:.10e}\n")
                        
                        if exact_value is not None:
                            abs_error, rel_error = calculate_error(rich_backward, exact_value)
                            self.results_text.insert(tk.END, f"    Error absoluto: {abs_error:.10e}\n")
                    except Exception as e:
                        self.results_text.insert(tk.END, f"  Error en extrapolación de regresivas: {str(e)}\n")
                
                if self.use_central.get():
                    try:
                        rich_central = richardson_extrapolation(self.function, x, h, 'central', order)
                        results['richardson_central'] = rich_central
                        self.results_text.insert(tk.END, f"  Centrales mejoradas: {rich_central:.10e}\n")
                        
                        if exact_value is not None:
                            abs_error, rel_error = calculate_error(rich_central, exact_value)
                            self.results_text.insert(tk.END, f"    Error absoluto: {abs_error:.10e}\n")
                    except Exception as e:
                        self.results_text.insert(tk.END, f"  Error en extrapolación de centrales: {str(e)}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            # Generar gráficos según la opción seleccionada
            self.generate_plots(results, x, h, order, exact_value)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el cálculo: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def find_optimal_h(self):
        """Encuentra el tamaño de paso óptimo para minimizar el error"""
        if not self.function or not self.exact_derivative_func:
            messagebox.showerror("Error", "Primero debe establecer una función válida")
            return
        
        try:
            x = float(self.x_value.get())
            order = self.order_value.get()
            
            self.results_text.insert(tk.END, "\nBúsqueda de tamaño de paso óptimo:\n")
            
            methods = []
            if self.use_forward.get():
                methods.append('forward')
            if self.use_backward.get():
                methods.append('backward')
            if self.use_central.get():
                methods.append('central')
            
            if not methods:
                messagebox.showinfo("Información", "Seleccione al menos un método de diferencias")
                return
            
            for method in methods:
                try:
                    optimal_h = optimal_step_size(
                        self.function, x, method, order, start_h=float(self.h_value.get()))
                    
                    self.results_text.insert(tk.END, f"  Método '{method}':\n")
                    self.results_text.insert(tk.END, f"    h óptimo: {optimal_h:.10e}\n")
                    
                    # Actualizar el valor de h en la interfaz si es el primer método
                    if method == methods[0]:
                        self.h_value.set(f"{optimal_h:.10g}")
                        
                except Exception as e:
                    self.results_text.insert(tk.END, f"  Error en método '{method}': {str(e)}\n")
            
            messagebox.showinfo("Información", "Tamaño de paso óptimo calculado")
        except Exception as e:
            messagebox.showerror("Error", f"Error al encontrar el paso óptimo: {str(e)}")
    
    def generate_plots(self, results, x, h, order, exact_value):
        """Genera los gráficos según la opción seleccionada"""
        plot_type = self.plot_type.get()
        
        # Limpiar figura existente
        self.fig.clear()
        
        try:
            if plot_type == "comparison":
                # Determinar los métodos seleccionados
                methods = []
                if self.use_forward.get():
                    methods.append('forward')
                if self.use_backward.get():
                    methods.append('backward')
                if self.use_central.get():
                    methods.append('central')
                
                if not methods:
                    messagebox.showinfo("Información", "Seleccione al menos un método para graficar")
                    return
                
                # Determinar el rango de x para el gráfico
                x_range = (x - 5*h, x + 5*h)
                
                # Generar gráfico de comparación
                fig = plot_derivatives(
                    self.function, x_range, methods, h, order, self.exact_derivative_func)
                
                # Copiar la figura al canvas
                for ax in fig.get_axes():
                    self.fig.add_axes(ax)
                
            elif plot_type == "error_h":
                if not self.exact_derivative_func:
                    messagebox.showerror("Error", "Se requiere la derivada exacta para este gráfico")
                    return
                
                # Determinar los métodos seleccionados
                methods = []
                if self.use_forward.get():
                    methods.append('forward')
                if self.use_backward.get():
                    methods.append('backward')
                if self.use_central.get():
                    methods.append('central')
                
                if not methods:
                    messagebox.showinfo("Información", "Seleccione al menos un método para graficar")
                    return
                
                # Generar gráfico de error vs h
                h_range = (1e-6, 1.0)
                fig = plot_error_vs_h(
                    self.function, x, methods, h_range, order, self.exact_derivative_func)
                
                # Copiar la figura al canvas
                for ax in fig.get_axes():
                    self.fig.add_axes(ax)
                
            elif plot_type == "function_deriv":
                # Determinar el método a utilizar
                method = None
                if self.use_central.get():
                    method = 'central'  # Preferir diferencias centrales
                elif self.use_forward.get():
                    method = 'forward'
                elif self.use_backward.get():
                    method = 'backward'
                
                if not method:
                    messagebox.showinfo("Información", "Seleccione al menos un método para graficar")
                    return
                
                # Determinar el rango de x para el gráfico
                x_range = (x - 5, x + 5)
                
                # Generar gráfico de función y derivada
                fig = plot_function_and_derivative(
                    self.function, x_range, method, h, order, self.exact_derivative_func)
                
                # Copiar la figura al canvas
                for ax in fig.get_axes():
                    self.fig.add_axes(ax)
            
            # Actualizar el canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar el gráfico: {str(e)}")
            import traceback
            traceback.print_exc()