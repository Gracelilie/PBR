import numpy as np
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.sampler import Lhs, Sobol
import math

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, Button, Select, Slider, Div, DataTable, TableColumn
from bokeh.palettes import Category10, Viridis
from bokeh.models.widgets import NumericInput
import threading
import time

# --- Configuration Constants ---
EST_SEC_PER_ITER = 2.5
WARN_THRESHOLD_ITERATIONS = 100
MAX_RECOMMENDED_ITERATIONS = 300

# --- Model Parameters ---
U_M = 0.152
U_D = 5.95*1e-3
K_N = 30.0*1e-3
Y_NX = 0.305
K_M = 0.350*1e-3*2
K_D = 3.71*0.05/90
K_NL = 10.0*1e-3
K_S = 142.8
K_I = 214.2
K_SL = 320.6
K_IL = 480.9
TAU = 0.120
KA = 0.0

# --- Global Variables ---
C_x0_model = 0.5
C_N0_model = 1.0
F_in_model = 8e-3
C_N_in_model = 10.0
I0_model = 150.0

# --- Data Sources for Bokeh ---
concentration_source = ColumnDataSource(data=dict(
    time=[],
    biomass=[],
    nitrate=[],
    lutein=[],
    lutein_scaled=[]
))

optimization_source = ColumnDataSource(data=dict(
    iteration=[],
    objective_value=[],
    best_objective=[],
    lutein_conc=[]
))

best_params_source = ColumnDataSource(data=dict(
    parameter=['C_x0', 'C_N0', 'F_in', 'C_N_in', 'I0'],
    value=[0.5, 1.0, 8e-3, 10.0, 150.0],
    optimal_value=[0.5, 1.0, 8e-3, 10.0, 150.0]
))

# --- ODE Model ---
def pbr(t, C):
    C_X = max(C[0], 1e-9)
    C_N = max(C[1], 1e-9)
    C_L = max(C[2], 1e-9)
    
    I = 2 * I0_model * (np.exp(-(TAU * 0.01 * 1000 * C_X)))
    Iscaling_u = I / (I + K_S + I**2 / K_I)
    Iscaling_k = I / (I + K_SL + I**2 / K_IL)
    u0 = U_M * Iscaling_u
    k0 = K_M * Iscaling_k

    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X
    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in_model * C_N_in_model
    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X
    
    return np.array([dCxdt, dCndt, dCldt])

def simulate_and_update_plot(C_x0, C_N0, F_in, C_N_in, I0):
    """Simulate the system and update the concentration plot"""
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model = C_x0
    C_N0_model = C_N0
    F_in_model = F_in
    C_N_in_model = C_N_in
    I0_model = I0

    time_span = [0, 150]
    time_eval = np.linspace(0, 150, 200)
    initial_conditions = np.array([C_x0_model, C_N0_model, 0.0])

    try:
        sol = solve_ivp(pbr, time_span, initial_conditions, t_eval=time_eval)
        
        concentration_source.data = dict(
            time=sol.t,
            biomass=sol.y[0],
            nitrate=sol.y[1],
            lutein=sol.y[2],
            lutein_scaled=sol.y[2] * 1000  # Scale for better visibility
        )
        
        return sol.y[2, -1] if len(sol.y[2]) > 0 else 0
    except Exception as e:
        print(f"Simulation error: {e}")
        return 0

def _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0):
    final_lutein = simulate_and_update_plot(C_x0, C_N0, F_in, C_N_in, I0)
    return -final_lutein if final_lutein > 0 else np.inf

# --- Optimization Setup ---
dimensions = [
    Real(0.2, 2.0, name='C_x0'),
    Real(0.2, 2.0, name='C_N0'),
    Real(1e-3, 1.5e-2, name='F_in'),
    Real(5.0, 15.0, name='C_N_in'),
    Real(100.0, 200.0, name='I0')
]

@use_named_args(dimensions)
def objective_function(C_x0, C_N0, F_in, C_N_in, I0):
    return _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0)

# --- Global optimization state ---
optimization_running = False
optimization_results = None

def run_optimization(surrogate_model, acquisition_func, n_iterations, initial_sampler, n_initial_points):
    """Run optimization in a separate thread"""
    global optimization_running, optimization_results
    
    optimization_running = True
    optimization_history = []
    
    def callback(res):
        current_best_lutein = -res.fun
        iteration_data = {
            'iteration': len(res.func_vals),
            'objective_value': -res.func_vals[-1],
            'best_objective_so_far': current_best_lutein,
            'current_params': res.x_iters[-1] if res.x_iters else None
        }
        optimization_history.append(iteration_data)
        
        # Update optimization plot
        iterations = [d['iteration'] for d in optimization_history]
        obj_vals = [d['objective_value'] for d in optimization_history]
        best_vals = [d['best_objective_so_far'] for d in optimization_history]
        
        optimization_source.data = dict(
            iteration=iterations,
            objective_value=obj_vals,
            best_objective=best_vals,
            lutein_conc=obj_vals
        )
        
        # Update best parameters if this is the best so far
        if iteration_data['current_params'] and current_best_lutein == max(best_vals):
            best_params_source.data['optimal_value'] = iteration_data['current_params']
            # Simulate with best parameters
            simulate_and_update_plot(*iteration_data['current_params'])

    # Generate initial points
    initial_points_x = []
    initial_points_y = []
    
    if n_initial_points > 0:
        np.random.seed(42)
        if initial_sampler == 'random':
            for _ in range(n_initial_points):
                point = [dim.rvs()[0] for dim in dimensions]
                initial_points_x.append(point)
        elif initial_sampler == 'lhs':
            sampler = Lhs(lhs_type="centered", criterion="maximin")
            initial_points_x = sampler.generate(dimensions, n_samples=n_initial_points, random_state=42)
        elif initial_sampler == 'sobol':
            sampler = Sobol()
            initial_points_x = sampler.generate(dimensions, n_samples=n_initial_points, random_state=42)
        
        for p_values_list in initial_points_x:
            initial_points_y.append(_evaluate_lutein_model_objective(*p_values_list))

    try:
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            base_estimator=surrogate_model,
            acq_func=acquisition_func,
            n_calls=n_iterations,
            x0=initial_points_x if n_initial_points > 0 else None,
            y0=initial_points_y if n_initial_points > 0 else None,
            n_initial_points=0 if n_initial_points > 0 else 10,
            random_state=42,
            callback=[callback]
        )
        
        optimization_results = result
        
        # Final update with optimal parameters
        final_lutein = simulate_and_update_plot(*result.x)
        best_params_source.data['optimal_value'] = result.x
        
        status_div.text = f"""
        <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px;'>
        <h3 style='color: #155724; margin-top: 0;'>Optimization Complete!</h3>
        <p><strong>Maximum Lutein Concentration:</strong> {-result.fun:.6f} g/L</p>
        <p><strong>Total Iterations:</strong> {len(optimization_history)}</p>
        </div>
        """
        
    except Exception as e:
        status_div.text = f"""
        <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;'>
        <h3 style='color: #721c24; margin-top: 0;'>Optimization Error</h3>
        <p>{str(e)}</p>
        </div>
        """
    
    optimization_running = False

# --- Bokeh Interface ---
# Create plots
concentration_plot = figure(
    title="Lutein Reactor Simulation - Concentration vs Time",
    x_axis_label='Time (h)',
    y_axis_label='Concentration (g/L)',
    width=800,
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

concentration_plot.line('time', 'biomass', source=concentration_source, 
                       line_width=2, color=Category10[3][0], legend_label="Biomass (C_X)")
concentration_plot.line('time', 'nitrate', source=concentration_source, 
                       line_width=2, color=Category10[3][1], legend_label="Nitrate (C_N)")
concentration_plot.line('time', 'lutein_scaled', source=concentration_source, 
                       line_width=2, color=Category10[3][2], legend_label="Lutein (C_L ×1000)")

concentration_plot.legend.location = "top_right"
concentration_plot.legend.click_policy = "hide"

# Optimization progress plot
optimization_plot = figure(
    title="Optimization Progress",
    x_axis_label='Iteration',
    y_axis_label='Lutein Concentration (g/L)',
    width=800,
    height=300,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

optimization_plot.line('iteration', 'objective_value', source=optimization_source,
                      line_width=2, color='blue', legend_label="Current Iteration")
optimization_plot.line('iteration', 'best_objective', source=optimization_source,
                      line_width=3, color='red', legend_label="Best So Far")

optimization_plot.legend.location = "bottom_right"

# Control widgets
surrogate_select = Select(title="Surrogate Model", value="GP", 
                         options=[("GP", "Gaussian Process"), ("RF", "Random Forest"), ("ET", "Extra Trees")])

acquisition_select = Select(title="Acquisition Function", value="gp_hedge",
                           options=[("gp_hedge", "GP Hedge"), ("EI", "Expected Improvement"), 
                                   ("PI", "Probability of Improvement"), ("LCB", "Lower Confidence Bound")])

iterations_input = NumericInput(title="Iterations", value=50, low=1, high=MAX_RECOMMENDED_ITERATIONS)

sampler_select = Select(title="Initial Sampling", value="random",
                       options=[("random", "Random"), ("lhs", "Latin Hypercube"), ("sobol", "Sobol")])

initial_points_input = NumericInput(title="Initial Points", value=10, low=1, high=50)

# Parameter sliders for manual testing
c_x0_slider = Slider(title="Initial Biomass (C_x0)", value=0.5, start=0.2, end=2.0, step=0.1)
c_n0_slider = Slider(title="Initial Nitrate (C_N0)", value=1.0, start=0.2, end=2.0, step=0.1)
f_in_slider = Slider(title="Flow Rate (F_in)", value=8e-3, start=1e-3, end=1.5e-2, step=1e-3)
c_n_in_slider = Slider(title="Inlet Nitrate (C_N_in)", value=10.0, start=5.0, end=15.0, step=0.5)
i0_slider = Slider(title="Light Intensity (I0)", value=150.0, start=100.0, end=200.0, step=10.0)

# Buttons
run_optimization_btn = Button(label="Run Optimization", button_type="success", width=200)
simulate_btn = Button(label="Simulate Current Parameters", button_type="primary", width=200)

# Status display
status_div = Div(text="""
<div style='padding: 10px; background-color: #e7f3ff; border: 1px solid #b8daff; border-radius: 5px;'>
<h3 style='color: #004085; margin-top: 0;'>Ready to Optimize</h3>
<p>Configure your optimization parameters and click 'Run Optimization' to begin.</p>
</div>
""", width=800)

# Parameters table
param_columns = [
    TableColumn(field="parameter", title="Parameter"),
    TableColumn(field="value", title="Current Value"),
    TableColumn(field="optimal_value", title="Optimal Value")
]
param_table = DataTable(source=best_params_source, columns=param_columns, width=400, height=200)

# Event handlers
def update_simulation():
    simulate_and_update_plot(
        c_x0_slider.value,
        c_n0_slider.value,
        f_in_slider.value,
        c_n_in_slider.value,
        i0_slider.value
    )
    # Update current values in table
    best_params_source.data['value'] = [
        c_x0_slider.value,
        c_n0_slider.value,
        f_in_slider.value,
        c_n_in_slider.value,
        i0_slider.value
    ]

def run_optimization_callback():
    if optimization_running:
        return
    
    status_div.text = """
    <div style='padding: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;'>
    <h3 style='color: #856404; margin-top: 0;'>Optimization Running...</h3>
    <p>Please wait while the optimization process is running. This may take several minutes.</p>
    </div>
    """
    
    # Clear previous results
    optimization_source.data = dict(iteration=[], objective_value=[], best_objective=[], lutein_conc=[])
    
    # Start optimization in a thread
    thread = threading.Thread(
        target=run_optimization,
        args=(
            surrogate_select.value,
            acquisition_select.value,
            int(iterations_input.value),
            sampler_select.value,
            int(initial_points_input.value)
        )
    )
    thread.daemon = True
    thread.start()

# Connect event handlers
for slider in [c_x0_slider, c_n0_slider, f_in_slider, c_n_in_slider, i0_slider]:
    slider.on_change('value', lambda attr, old, new: update_simulation())

run_optimization_btn.on_click(run_optimization_callback)
simulate_btn.on_click(update_simulation)

# Initial simulation
update_simulation()

# Layout
controls = column(
    Div(text="<h2>Optimization Parameters</h2>"),
    row(surrogate_select, acquisition_select),
    row(iterations_input, initial_points_input),
    sampler_select,
    row(run_optimization_btn, simulate_btn),
    status_div
)

manual_controls = column(
    Div(text="<h2>Manual Parameter Testing</h2>"),
    c_x0_slider,
    c_n0_slider,
    f_in_slider,
    c_n_in_slider,
    i0_slider
)

results_panel = column(
    Div(text="<h2>Optimization Results</h2>"),
    param_table
)

# Main layout
layout = column(
    Div(text="<h1>Interactive Lutein Production Optimization</h1>"),
    row(controls, manual_controls, results_panel),
    concentration_plot,
    optimization_plot
)

# Add to document
curdoc().add_root(layout)
curdoc().title = "Lutein Production Optimizer"