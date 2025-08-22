import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Tuple, Callable, Optional
import sympy as sp
import re
import json

from lagrange_core import lagrange_interpolation, get_all_basis_values
from error_calculator import local_error, global_error, error_bound
from plotter import plot_interpolation, plot_error

class InterpolationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Interpolación de Lagrange")
        self.geometry("1200x800")
        self.configure(padx=10, pady=10)
        
        self.x_points = []
        self.y_points = []
        self.original_func = None
        self.derivative_func = None
        
        self.create_variables()
        self.create_widgets()
        self.setup_layout()
    
    def create_variables(self):
        """Inicializa variables del programa"""
        self.func_str = tk.StringVar(value="")
        self.derivative_str = tk.StringVar(value="")
        self.point_x = tk.StringVar(value="")
        self.point_y = tk.StringVar(value="")
        self.show_basis = tk.BooleanVar(value=False)
        self.show_original = tk.BooleanVar(value=True)
        self.show_error_plot = tk.BooleanVar(value=False)
        self.test_function_var = tk.StringVar(value="sin")  # Nueva variable para funciones de prueba
    
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
        # Frame para puntos
        self.points_frame = ttk.LabelFrame(self.control_frame, text="Puntos de interpolación", padding=5)
        
        # Entrada de puntos
        ttk.Label(self.points_frame, text="x:").grid(row=0, column=0, padx=5, pady=5)
        self.x_entry = ttk.Entry(self.points_frame, textvariable=self.point_x, width=10)
        self.x_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.points_frame, text="y:").grid(row=0, column=2, padx=5, pady=5)
        self.y_entry = ttk.Entry(self.points_frame, textvariable=self.point_y, width=10)
        self.y_entry.grid(row=0, column=3, padx=5, pady=5)
        
        self.add_point_btn = ttk.Button(self.points_frame, text="Agregar punto", command=self.add_point)
        self.add_point_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Lista de puntos
        self.points_list_frame = ttk.Frame(self.points_frame)
        self.points_list_frame.grid(row=1, column=0, columnspan=5, padx=5, pady=5, sticky="nsew")
        
        self.points_listbox = tk.Listbox(self.points_list_frame, height=10, width=30)
        self.points_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.points_scrollbar = ttk.Scrollbar(self.points_list_frame, orient=tk.VERTICAL, 
                                             command=self.points_listbox.yview)
        self.points_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.points_listbox.config(yscrollcommand=self.points_scrollbar.set)
        
        # Botones para manipular puntos
        self.points_btn_frame = ttk.Frame(self.points_frame)
        self.points_btn_frame.grid(row=2, column=0, columnspan=5, padx=5, pady=5)
        
        self.remove_point_btn = ttk.Button(self.points_btn_frame, text="Eliminar punto", 
                                          command=self.remove_point)
        self.remove_point_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_points_btn = ttk.Button(self.points_btn_frame, text="Limpiar todo", 
                                          command=self.clear_points)
        self.clear_points_btn.pack(side=tk.LEFT, padx=5)
        
        # NUEVO: Generador de puntos de prueba
        self.test_frame = ttk.LabelFrame(self.control_frame, text="Datos de prueba", padding=5)
        
        ttk.Label(self.test_frame, text="Función:").grid(row=0, column=0, padx=5, pady=5)
        self.test_function_combo = ttk.Combobox(self.test_frame, textvariable=self.test_function_var,
                                               values=["sin", "exp", "poly"])
        self.test_function_combo.grid(row=0, column=1, padx=5, pady=5)
        
        self.generate_test_btn = ttk.Button(self.test_frame, text="Generar puntos de prueba",
                                           command=self.generate_test_points)
        self.generate_test_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Frame para función original
        self.func_frame = ttk.LabelFrame(self.control_frame, text="Función original (opcional)", padding=5)
        
        ttk.Label(self.func_frame, text="f(x) = ").grid(row=0, column=0, padx=5, pady=5)
        self.func_entry = ttk.Entry(self.func_frame, textvariable=self.func_str, width=30)
        self.func_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.func_frame, text="f'(x) = ").grid(row=1, column=0, padx=5, pady=5)
        self.derivative_entry = ttk.Entry(self.func_frame, textvariable=self.derivative_str, width=30)
        self.derivative_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.set_func_btn = ttk.Button(self.func_frame, text="Establecer función", 
                                      command=self.set_function)
        self.set_func_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Label(self.func_frame, text="Ejemplos: sin(x), x**2, exp(-x), x**3 - 2*x + 1").grid(
            row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Frame para opciones de visualización
        self.viz_frame = ttk.LabelFrame(self.control_frame, text="Opciones de visualización", padding=5)
        
        self.show_basis_check = ttk.Checkbutton(self.viz_frame, text="Mostrar bases de Lagrange", 
                                               variable=self.show_basis)
        self.show_basis_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.show_original_check = ttk.Checkbutton(self.viz_frame, text="Mostrar función original", 
                                                 variable=self.show_original)
        self.show_original_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.show_error_check = ttk.Checkbutton(self.viz_frame, text="Mostrar gráfico de error", 
                                               variable=self.show_error_plot)
        self.show_error_check.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # Botones de acción
        self.action_frame = ttk.Frame(self.control_frame, padding=5)
        
        self.interpolate_btn = ttk.Button(self.action_frame, text="Interpolar", 
                                         command=self.interpolate)
        self.interpolate_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.save_btn = ttk.Button(self.action_frame, text="Guardar datos", 
                                  command=self.save_data)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.load_btn = ttk.Button(self.action_frame, text="Cargar datos", 
                                  command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # NUEVO: Botón de salir
        self.exit_btn = ttk.Button(self.action_frame, text="Salir", 
                                  command=self.quit)
        self.exit_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Frame para resultados
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Resultados", padding=5)
        
        self.results_text = tk.Text(self.results_frame, height=10, width=40, wrap=tk.WORD)
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
        self.ax.set_title("Esperando datos para interpolar...")
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
        self.points_frame.pack(fill=tk.X, pady=5)
        self.test_frame.pack(fill=tk.X, pady=5)  # Nuevo frame de datos de prueba
        self.func_frame.pack(fill=tk.X, pady=5)
        self.viz_frame.pack(fill=tk.X, pady=5)
        self.action_frame.pack(fill=tk.X, pady=5)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def add_point(self):
        """Añade un punto a la lista de puntos"""
        try:
            x = float(self.point_x.get())
            y = float(self.point_y.get())
            
            # Verificar si el punto ya existe o está muy cercano a otro
            for existing_x in self.x_points:
                if abs(x - existing_x) < 1e-10:  # Tolerancia numérica
                    idx = self.x_points.index(existing_x)
                    self.y_points[idx] = y
                    self.update_points_list()
                    messagebox.showinfo("Información", f"Se actualizó el punto ({existing_x}, {y})")
                    
                    # Limpiar entradas
                    self.point_x.set("")
                    self.point_y.set("")
                    self.x_entry.focus()
                    return
            
            # Verificar valores no finitos
            if not np.isfinite(x) or not np.isfinite(y):
                messagebox.showerror("Error", "Los valores no son números válidos")
                return
                
            self.x_points.append(x)
            self.y_points.append(y)
            self.update_points_list()
            
            # Limpiar entradas
            self.point_x.set("")
            self.point_y.set("")
            self.x_entry.focus()
        except ValueError:
            messagebox.showerror("Error", "Los valores de x e y deben ser números válidos")
    
    def update_points_list(self):
        """Actualiza la lista de puntos mostrada en el listbox"""
        # Ordenar puntos por valor de x
        points = sorted(zip(self.x_points, self.y_points))
        self.x_points = [p[0] for p in points]
        self.y_points = [p[1] for p in points]
        
        # Actualizar listbox
        self.points_listbox.delete(0, tk.END)
        for x, y in points:
            self.points_listbox.insert(tk.END, f"({x}, {y})")
    
    def remove_point(self):
        """Elimina el punto seleccionado"""
        try:
            idx = self.points_listbox.curselection()[0]
            del self.x_points[idx]
            del self.y_points[idx]
            self.update_points_list()
        except (IndexError, ValueError):
            messagebox.showerror("Error", "Seleccione un punto para eliminar")
    
    def clear_points(self):
        """Elimina todos los puntos"""
        self.x_points = []
        self.y_points = []
        self.points_listbox.delete(0, tk.END)
    
    def generate_test_points(self):
        """Genera puntos de prueba para funciones comunes"""
        function_type = self.test_function_var.get()
        
        # Limpiar puntos actuales
        self.clear_points()
        
        if function_type == "sin":
            # Generar puntos para sin(x)
            x_points = np.linspace(0, 2*np.pi, 5)
            for x in x_points:
                self.x_points.append(float(x))
                self.y_points.append(float(np.sin(x)))
            self.func_str.set("sin(x)")
            
        elif function_type == "exp":
            # Generar puntos para exp(x)
            x_points = np.linspace(-2, 2, 5)
            for x in x_points:
                self.x_points.append(float(x))
                self.y_points.append(float(np.exp(x)))
            self.func_str.set("exp(x)")
            
        elif function_type == "poly":
            # Generar puntos para x^3 - 2*x + 1
            x_points = np.linspace(-2, 2, 5)
            for x in x_points:
                self.x_points.append(float(x))
                self.y_points.append(float(x**3 - 2*x + 1))
            self.func_str.set("x**3 - 2*x + 1")
        
        # Actualizar la lista de puntos y establecer la función
        self.update_points_list()
        self.set_function()
    
    def set_function(self):
        """Establece la función original y su derivada"""
        try:
            func_str = self.func_str.get().strip()
            deriv_str = self.derivative_str.get().strip()
            
            if not func_str:
                self.original_func = None
                self.derivative_func = None
                messagebox.showinfo("Información", "Función eliminada")
                return
            
            # Crear función con sympy y convertir a función numérica
            x = sp.Symbol('x')
            
            # Limpiar las expresiones de funciones comunes
            func_str = func_str.replace("^", "**")
            if deriv_str:
                deriv_str = deriv_str.replace("^", "**")
            
            # Crear expresiones simbólicas
            try:
                func_expr = sp.sympify(func_str)
            except:
                messagebox.showerror("Error", "No se pudo interpretar la función. Verifique la sintaxis.")
                return
            
            # Si no se proporciona la derivada, calcularla
            if not deriv_str:
                # Calcular la derivada de orden adecuado para la cota de error
                if len(self.x_points) > 0:
                    n = len(self.x_points)
                    # Calcular la derivada de orden n+1
                    deriv_expr = func_expr
                    for i in range(n+1):
                        deriv_expr = sp.diff(deriv_expr, x)
                else:
                    # Si no hay puntos, calcular solo la primera derivada
                    deriv_expr = sp.diff(func_expr, x)
                    
                self.derivative_str.set(str(deriv_expr))
            else:
                try:
                    deriv_expr = sp.sympify(deriv_str)
                except:
                    messagebox.showerror("Error", "No se pudo interpretar la derivada. Verifique la sintaxis.")
                    return
            
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
            
            # Asignar las funciones seguras
            self.original_func = safe_func
            self.derivative_func = safe_deriv
            
            messagebox.showinfo("Información", "Función establecida correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al establecer la función: {str(e)}")
            self.original_func = None
            self.derivative_func = None
    
    def interpolate(self):
        """Realiza la interpolación y muestra los resultados"""
        if len(self.x_points) < 2:
            messagebox.showerror("Error", "Se necesitan al menos 2 puntos para interpolar")
            return
        
        # Verificar que todos los puntos x sean distintos
        if len(set(self.x_points)) != len(self.x_points):
            messagebox.showerror("Error", "Todos los valores de x deben ser distintos para la interpolación")
            return
            
        try:
            # Limpiar resultados anteriores
            self.results_text.delete(1.0, tk.END)
            
            # Crear figuras según opciones seleccionadas
            self.fig.clear()
            
            if self.show_error_plot.get() and self.original_func:
                # Crear dos subplots: interpolación y error
                gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1])
                ax_interp = self.fig.add_subplot(gs[0])
                ax_error = self.fig.add_subplot(gs[1])
                
                # Graficar interpolación
                x_min, x_max = min(self.x_points) - 0.5, max(self.x_points) + 0.5
                eval_points = np.linspace(x_min, x_max, 1000)
                
                # Puntos de interpolación
                ax_interp.scatter(self.x_points, self.y_points, color='red', zorder=5, s=50, 
                                label='Puntos de interpolación')
                
                # Polinomio de Lagrange
                interp_values = [lagrange_interpolation(x, self.x_points, self.y_points) 
                                for x in eval_points]
                ax_interp.plot(eval_points, interp_values, 'b-', linewidth=2, 
                              label='Polinomio de Lagrange')
                
                # Función original si está disponible y se seleccionó
                if self.original_func and self.show_original.get():
                    # Manejo seguro de evaluación de funciones
                    original_values = []
                    valid_points = []
                    
                    for x in eval_points:
                        try:
                            value = self.original_func(x)
                            # Verificar si el valor es válido (no NaN, no infinito)
                            if np.isfinite(value):
                                original_values.append(value)
                                valid_points.append(x)
                        except:
                            # Ignorar puntos donde la función no se puede evaluar
                            continue
                    
                    if valid_points:
                        ax_interp.plot(valid_points, original_values, 'g--', linewidth=1.5, 
                                      label='Función original')
                
                # Bases de Lagrange si se seleccionó
                if self.show_basis.get():
                    basis_values = get_all_basis_values(self.x_points, eval_points)
                    for i, basis in enumerate(basis_values):
                        ax_interp.plot(eval_points, basis, '--', linewidth=0.7, alpha=0.6, 
                                      label=f'Base L_{i}(x)')
                
                # Configurar gráfico de interpolación
                ax_interp.legend(loc='best')
                ax_interp.set_xlabel('x')
                ax_interp.set_ylabel('y')
                ax_interp.set_title('Interpolación de Lagrange')
                ax_interp.grid(True, alpha=0.3)
                
                # Graficar error
                if self.original_func:
                    # Calcular errores con manejo seguro de funciones
                    valid_points = []
                    errors = []
                    
                    for x in eval_points:
                        try:
                            interp_value = lagrange_interpolation(x, self.x_points, self.y_points)
                            orig_value = self.original_func(x)
                            
                            # Verificar valores válidos
                            if np.isfinite(interp_value) and np.isfinite(orig_value):
                                valid_points.append(x)
                                errors.append(abs(interp_value - orig_value))
                        except:
                            continue
                    
                    # Graficar el error
                    if valid_points:
                        ax_error.plot(valid_points, errors, 'r-', linewidth=1.5, label='Error absoluto')
                    
                    # Marcar los puntos de interpolación
                    ax_error.scatter(self.x_points, [0] * len(self.x_points), color='blue', zorder=5, s=30, 
                                    label='Puntos de interpolación')
                    
                    ax_error.legend(loc='best')
                    ax_error.set_xlabel('x')
                    ax_error.set_ylabel('Error absoluto')
                    ax_error.set_title('Error de interpolación')
                    ax_error.grid(True, alpha=0.3)
            else:
                # Solo graficar interpolación
                ax = self.fig.add_subplot(111)
                
                # Graficar interpolación
                x_min, x_max = min(self.x_points) - 0.5, max(self.x_points) + 0.5
                eval_points = np.linspace(x_min, x_max, 1000)
                
                # Puntos de interpolación
                ax.scatter(self.x_points, self.y_points, color='red', zorder=5, s=50, 
                          label='Puntos de interpolación')
                
                # Polinomio de Lagrange
                interp_values = [lagrange_interpolation(x, self.x_points, self.y_points) 
                                for x in eval_points]
                ax.plot(eval_points, interp_values, 'b-', linewidth=2, 
                       label='Polinomio de Lagrange')
                
                # Función original si está disponible y se seleccionó
                if self.original_func and self.show_original.get():
                    # Manejo seguro de evaluación de funciones
                    original_values = []
                    valid_points = []
                    
                    for x in eval_points:
                        try:
                            value = self.original_func(x)
                            # Verificar si el valor es válido (no NaN, no infinito)
                            if np.isfinite(value):
                                original_values.append(value)
                                valid_points.append(x)
                        except:
                            # Ignorar puntos donde la función no se puede evaluar
                            continue
                    
                    if valid_points:
                        ax.plot(valid_points, original_values, 'g--', linewidth=1.5, 
                               label='Función original')
                
                # Bases de Lagrange si se seleccionó
                if self.show_basis.get():
                    basis_values = get_all_basis_values(self.x_points, eval_points)
                    for i, basis in enumerate(basis_values):
                        ax.plot(eval_points, basis, '--', linewidth=0.7, alpha=0.6, 
                               label=f'Base L_{i}(x)')
                
                # Configurar gráfico
                ax.legend(loc='best')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Interpolación de Lagrange')
                ax.grid(True, alpha=0.3)
            
            # Ajustar layout y actualizar canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Calcular errores y mostrar en resultados
            self.display_results(eval_points)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la interpolación: {str(e)}")
            import traceback
            traceback.print_exc()  # Mostrar detalles del error en la consola
    
    def display_results(self, eval_points):
        """Muestra los resultados de la interpolación en el panel de resultados"""
        # Mostrar grado del polinomio
        n = len(self.x_points) - 1
        self.results_text.insert(tk.END, f"Polinomio de grado: {n}\n\n")
        
        # Mostrar valores de las bases en cada punto de interpolación
        self.results_text.insert(tk.END, "Bases de Lagrange en puntos de interpolación:\n")
        for i, x_i in enumerate(self.x_points):
            self.results_text.insert(tk.END, f"Punto x_{i} = {x_i}:\n")
            for j in range(len(self.x_points)):
                basis_value = 1.0 if i == j else 0.0  # Propiedad de las bases de Lagrange
                self.results_text.insert(tk.END, f"  L_{j}({x_i}) = {basis_value:.10f}\n")
            self.results_text.insert(tk.END, "\n")
        
        # Mostrar errores si hay función original
        if self.original_func:
            self.results_text.insert(tk.END, "Errores de interpolación:\n")
            
            # Errores locales en los puntos de interpolación
            self.results_text.insert(tk.END, "Errores locales en puntos de interpolación:\n")
            for i, x_i in enumerate(self.x_points):
                try:
                    error = local_error(x_i, self.x_points, self.y_points, self.original_func)
                    if error is not None and np.isfinite(error):
                        self.results_text.insert(tk.END, f"  Error en x_{i} = {x_i}: {error:.10e}\n")
                    else:
                        self.results_text.insert(tk.END, f"  Error en x_{i} = {x_i}: No calculable\n")
                except:
                    self.results_text.insert(tk.END, f"  Error en x_{i} = {x_i}: Error en cálculo\n")
            
            # Error global
            try:
                # Filtrar puntos donde se pueda evaluar la función original
                valid_eval_points = []
                for x in eval_points:
                    try:
                        if np.isfinite(self.original_func(x)):
                            valid_eval_points.append(x)
                    except:
                        continue
                
                if valid_eval_points:
                    self.results_text.insert(tk.END, "\nError global:\n")
                    max_error, avg_error, mse = global_error(
                        self.x_points, self.y_points, self.original_func, valid_eval_points)
                    
                    if np.isfinite(max_error):
                        self.results_text.insert(tk.END, f"  Error máximo: {max_error:.10e}\n")
                    if np.isfinite(avg_error):
                        self.results_text.insert(tk.END, f"  Error promedio: {avg_error:.10e}\n")
                    if np.isfinite(mse):
                        self.results_text.insert(tk.END, f"  Error cuadrático medio: {mse:.10e}\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"\nError al calcular errores globales: {str(e)}\n")
            
            # Cota del error si hay derivada
            if self.derivative_func:
                try:
                    self.results_text.insert(tk.END, "\nCota teórica del error:\n")
                    # Seleccionar algunos puntos para mostrar la cota
                    test_points = np.linspace(min(eval_points), max(eval_points), 5)
                    for x in test_points:
                        try:
                            bound = error_bound(x, self.x_points, self.derivative_func)
                            if np.isfinite(bound):
                                self.results_text.insert(tk.END, f"  Cota en x = {x:.4f}: {bound:.10e}\n")
                            else:
                                self.results_text.insert(tk.END, f"  Cota en x = {x:.4f}: No calculable\n")
                        except Exception as e:
                            self.results_text.insert(tk.END, f"  Error al calcular cota en x = {x}: {str(e)}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"\nError al calcular cotas de error: {str(e)}\n")
    
    def save_data(self):
        """Guarda los datos actuales en un archivo JSON"""
        if not self.x_points:
            messagebox.showerror("Error", "No hay datos para guardar")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Guardar datos de interpolación"
            )
            
            if not filename:
                return
            
            # Preparar datos para guardar
            data = {
                "x_points": self.x_points,
                "y_points": self.y_points,
                "function": self.func_str.get(),
                "derivative": self.derivative_str.get()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            messagebox.showinfo("Información", f"Datos guardados en {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar los datos: {str(e)}")
    
    def load_data(self):
        """Carga datos desde un archivo JSON"""
        try:
            filename = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Cargar datos de interpolación"
            )
            
            if not filename:
                return
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Cargar datos
            self.x_points = data.get("x_points", [])
            self.y_points = data.get("y_points", [])
            self.func_str.set(data.get("function", ""))
            self.derivative_str.set(data.get("derivative", ""))
            
            # Actualizar interfaz
            self.update_points_list()
            if self.func_str.get():
                self.set_function()
            
            messagebox.showinfo("Información", f"Datos cargados desde {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos: {str(e)}")