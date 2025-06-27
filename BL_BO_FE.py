# # --- Add these new global variables at the top with other globals ---
# optimization_objective = "max_lutein"  # Can be "max_lutein" or "min_cost"
# target_lutein_mass = 1.0  # Default target mass in g
# power_consumption = 14.0  # Watts
# electricity_cost = 0.15  # $/kWh
# nitrogen_cost = 85.0  # $/1000mg = $85/kg
# biomass_cost = 8.0  # $/mL (assuming this is for concentrated stock)
# time_given = 150 # time it takes in hours for a total concentration 

# # # --- Add these new helper functions ---
# # def calculate_required_time(target_mass):
# #     """Calculate the time required to produce the target mass of lutein."""
# #     global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    
# #     # Run simulation to find when lutein reaches target mass
# #     def mass_reached(t, C):
# #         C_L = C[2]
# #         current_mass = C_L * 1.0  # Assuming 1L volume
# #         return target_mass - current_mass
    
#     mass_reached.terminal = True
    
#     # try:
#     #     sol = solve_ivp(
#     #         pbr, 
#     #         [0, 150], 
#     #         [C_x0_model, C_N0_model, 0.0], 
#     #         events=mass_reached,
#     #         method="RK45",
#     #         dense_output=True
#     #     )
        
#     #     if sol.status == 1:  # Event occurred
#     #         return sol.t_events[0][0]
#     #     else:
#     #         return 150.0  # Return max time if target not reached
#     # except:
#     #     return 150.0  # Return max time if solver fails

# def calculate_total_cost(params):
#     """Calculate the total cost based on the given parameters."""
#     global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
#     C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = params
    
#     # Calculate required time to produce target lutein mass
#     time_hours = calculate_required_time(target_lutein_mass) / 3600  # Convert to hours
    
#     # Cost components
#     nitrogen_operating_cost = (nitrogen_cost * C_N_in_model * F_in_model * time_hours) / 1000  # Convert mg to g
#     nitrogen_initial_cost = (nitrogen_cost * C_N0_model * 1.0) / 1000  # Assuming 1L volume
#     electricity_cost_total = (power_consumption * time_hours * electricity_cost) / 1000  # Convert W to kW
#     biomass_cost_total = biomass_cost * C_x0_model * 1.0  # Assuming 1L volume
    
#     total_cost = (nitrogen_operating_cost + nitrogen_initial_cost + 
#                  electricity_cost_total + biomass_cost_total)
    
#     return total_cost

# def _evaluate_objective(*args):
#     """Wrapper function that evaluates either lutein production or cost based on user selection."""
#     if optimization_objective == "max_lutein":
#         return _evaluate_lutein_model_objective(*args)
#     else:
#         total_cost = calculate_total_cost(args)
#         return total_cost  # We want to minimize this

# # --- Add these new UI widgets ---
# objective_select = Select(
#     title="Optimization Objective:", 
#     value="max_lutein", 
#     options=["max_lutein", "min_cost"],
#     width=150
# )

# target_lutein_input = Spinner(
#     title="Target Lutein Mass (g):", 
#     low=0.01, 
#     high=100.0, 
#     step=0.1, 
#     value=1.0,
#     width=150
# )

# # Add callback functions for the new widgets
# def update_optimization_objective(attr, old, new):
#     global optimization_objective
#     optimization_objective = new
#     update_status(f"Optimization objective set to: {new}")

# def update_target_lutein(attr, old, new):
#     global target_lutein_mass
#     target_lutein_mass = new

# objective_select.on_change('value', update_optimization_objective)
# target_lutein_input.on_change('value', update_target_lutein)

# # --- Modify the UI layout to include the new widgets ---
# # Add these to your existing controls_col definition:
# controls_col = column(
#     title_div, description_p,
#     param_range_title, cx0_range, cn0_range, fin_range, cnin_range, i0_range,
#     settings_title, surrogate_select, acq_func_select, sampler_select, n_initial_input,
#     Div(text="<h4>Optimization Target</h4>"),
#     row(objective_select, target_lutein_input),
#     actions_title, generate_button, calculate_button, suggest_button, suggestion_div, run_suggestion_button, reset_button,
#     status_div,
#     width=470,
# )

# # --- Update the results display to show cost information when relevant ---
# def process_and_plot_latest_results():
#     """Finds the best result from the history and updates plots."""
#     if not optimization_history: return
    
#     best_item = min(optimization_history, key=lambda item: item[1])
#     best_params, best_obj_val = best_item[0], best_item[1]
    
#     if optimization_objective == "max_lutein":
#         max_lutein = -best_obj_val
#         optimal_params = {dim.name: val for dim, val in zip(get_current_dimensions(), best_params)}

#         results_html = f"<h3>Overall Best Result So Far</h3>"
#         results_html += f"<b>Maximum Lutein Found:</b> {max_lutein:.4f} g/L<br/>"
#         results_html += "<b>Corresponding Parameters:</b><ul>"
#         for param, value in optimal_params.items(): results_html += f"<li><b>{param}:</b> {value:.4f}</li>"
#         results_html += "</ul>"
#     else:
#         min_cost = best_obj_val
#         optimal_params = {dim.name: val for dim, val in zip(get_current_dimensions(), best_params)}
        
#         # Calculate time required
#         time_hours = calculate_required_time(target_lutein_mass) / 3600
        
#         results_html = f"<h3>Overall Best Result So Far</h3>"
#         results_html += f"<b>Minimum Cost Found:</b> ${min_cost:.2f}<br/>"
#         results_html += f"<b>Time Required:</b> {time_hours:.2f} hours<br/>"
#         results_html += "<b>Corresponding Parameters:</b><ul>"
#         for param, value in optimal_params.items(): results_html += f"<li><b>{param}:</b> {value:.4f}</li>"
#         results_html += "</ul>"
        
#     results_div.text = results_html
    
#     update_convergence_plot_from_history()
#     run_final_simulation(best_params)






#    I cannot directly execute the Bokeh application in this environment. However, I can provide you with the complete modified code for `BO_test.py` that includes the option to optimize the yield of lutein.

# Here is the updated `BO_test.py` code with the yield optimization functionality:

# ```python
import numpy as np
from scipy.integrate import solve_ivp
from skopt import Optimizer
from skopt.space import Real
from skopt.sampler import Lhs, Sobol
import threading

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Button,
    Select,
    Div,
    Spinner,
    Paragraph,
    DataTable,
    TableColumn,
    NumberFormatter,
    RangeSlider,
    LinearAxis,
    DataRange1d,
    NumeralTickFormatter,
    FixedTicker,
)

# --- 1. Define Model Parameters (Constants of the Lutein system) ---
U_M = 0.152
U_D = 5.95e-3
K_N = 30.0e-3
Y_NX = 0.305
K_M = 0.350e-3 * 2
K_D = 3.71 * 0.05 / 90
K_NL = 10.0e-3
K_S = 142.8
K_I = 214.2
K_SL = 320.6
K_IL = 480.9
TAU = 0.120
KA = 0.0

# --- Cost and Process Constants ---
TIME_HOURS = 150  # hours
VOLUME_L = 500  # L
DENSITY_G_PER_L = 1  # g/L
COST_N_PER_G = 0.005  # $/g for nitrogen
COST_BIOMASS_PER_V = 2.00  # $/V for biomass
LUTEIN_PRICE_PER_MG = 0.20  # $/mg for lutein

# --- New Constant for Lutein Yield Calculation ---
THEORETICAL_LUTEIN_PER_BIOMASS = 0.00556 # g Lu / 1 g Biomass

# --- Global Variables for ODE solver ---
C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = 0.5, 1.0, 8e-3, 10.0, 150.0

# --- Global state for the interactive experiment ---
optimization_history = [] # Stores all evaluated points [x_params, y_value]
optimizer = None # The scikit-optimize Optimizer object
previewed_point = None # Stores the point from the "Preview" action
optimization_mode = "concentration"  # "concentration", "cost", or "yield"

# --- 2. Define the Photobioreactor ODE Model ---
def pbr(t, C):
    """Defines the system of Ordinary Differential Equations for the photobioreactor."""
    C_X, C_N, C_L = C
    if C_X < 1e-9: C_X = 1e-9
    if C_N < 1e-9: C_N = 1e-9
    if C_L < 1e-9: C_L = 1e-9

    I = 2 * I0_model * (np.exp(-(TAU * 0.01 * 1000 * C_X)))
    Iscaling_u = I / (I + K_S + I ** 2 / K_I)
    Iscaling_k = I / (I + K_SL + I ** 2 / K_IL)
    u0 = U_M * Iscaling_u
    k0 = K_M * Iscaling_k

    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X
    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in_model * C_N_in_model
    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X
    return np.array([dCxdt, dCndt, dCldt])

# --- 3. Cost Calculation Functions ---
def calculate_nitrogen_cost(C_N0, C_N_in, F_in):
    """Calculate nitrogen cost based on initial and feed concentrations"""
    # Cost from feed nitrogen: CostN * C_N_in * F_in * time
    feed_cost = COST_N_PER_G * C_N_in * F_in * TIME_HOURS
    # Cost from initial nitrogen: CostN * C_N0 * V
    initial_cost = COST_N_PER_G * C_N0 * VOLUME_L
    return feed_cost + initial_cost

def calculate_biomass_cost(C_x0):
    """Calculate biomass cost based on initial concentration"""
    # Cost: $2.00/V * density * C_x0 * V
    return COST_BIOMASS_PER_V * DENSITY_G_PER_L * C_x0 * VOLUME_L

def calculate_lutein_profit(lutein_concentration):
    """Calculate profit from lutein production"""
    # Convert g/L to mg/L (multiply by 1000)
    lutein_mg_per_L = lutein_concentration * 1000
    # Profit: price_per_mg * concentration_mg_per_L * volume_L
    return LUTEIN_PRICE_PER_MG * lutein_mg_per_L * VOLUME_L

def calculate_total_cost_and_profit(params, lutein_concentration):
    """Calculate total cost, profit, and revenue (J)"""
    C_x0, C_N0, F_in, C_N_in, I0 = params
    
    nitrogen_cost = calculate_nitrogen_cost(C_N0, C_N_in, F_in)
    biomass_cost = calculate_biomass_cost(C_x0)
    total_cost = nitrogen_cost + biomass_cost
    
    lutein_profit = calculate_lutein_profit(lutein_concentration)
    
    # J = total_cost - profit (we want to minimize J, which maximizes profit - cost)
    J = total_cost - lutein_profit
    
    return {
        'nitrogen_cost': nitrogen_cost,
        'biomass_cost': biomass_cost,
        'total_cost': total_cost,
        'lutein_profit': lutein_profit,
        'revenue_J': J
    }

# --- New Yield Calculation Function ---
def calculate_lutein_yield(C_x_final, C_L_final):
    """
    Calculate the yield of lutein.
    Yield = (actual concentration of Lutein / theoretical lutein concentration) * 100
    Theoretical lutein concentration = Theoretical mass of lutein per 1g * Total Lutein Concentration
    Total Lutein Concentration = C_x_final + C_L_final
    """
    total_lutein_concentration_at_end = C_x_final + C_L_final
    theoretical_lutein_concentration = THEORETICAL_LUTEIN_PER_BIOMASS * total_lutein_concentration_at_end
    
    if theoretical_lutein_concentration <= 0:
        return 0.0 # Avoid division by zero
    
    yield_percent = (C_L_final / theoretical_lutein_concentration) * 100
    return yield_percent

# --- 4. Helper function to evaluate the model and objective ---
def _evaluate_lutein_model_objective(*args):
    """Sets up and runs a single simulation to find the final lutein concentration."""
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model, optimization_mode
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = args
    
    sol = solve_ivp(pbr, [0, 150], [C_x0_model, C_N0_model, 0.0], t_eval=[150], method="RK45")
    final_lutein = sol.y[2, -1]
    final_biomass = sol.y[0, -1] # Get final biomass concentration

    if not np.isfinite(final_lutein) or final_lutein <= 0:
        return 1e6  # Penalty for non-physical results
    
    if optimization_mode == "concentration":
        return -final_lutein  # Return negative because optimizer minimizes
    elif optimization_mode == "cost":
        cost_analysis = calculate_total_cost_and_profit(args, final_lutein)
        return cost_analysis['revenue_J']  # Minimize cost - profit
    else: # optimization_mode == "yield"
        lutein_yield = calculate_lutein_yield(final_biomass, final_lutein)
        return -lutein_yield # Maximize yield by minimizing negative yield

# --- 5. Bokeh Application Setup ---
doc = curdoc()
doc.title = "Lutein Production Optimizer"

# --- Data Sources ---
convergence_source = ColumnDataSource(data=dict(iter=[], best_value=[]))
simulation_source = ColumnDataSource(data=dict(time=[], C_X=[], C_N=[], C_L=[], C_L_scaled=[]))
experiments_source = ColumnDataSource(data=dict(
    C_x0=[], C_N0=[], F_in=[], C_N_in=[], I0=[], 
    Lutein=[], Total_Cost=[], Lutein_Profit=[], Revenue_J=[],
    Lutein_Yield=[] # Add new column for Lutein Yield
))

# --- UI and Workflow Functions ---
def set_optimization_mode():
    """Update optimization mode based on user selection and toggle cost/yield visibility."""
    global optimization_mode, optimizer
    new_mode = objective_select.value
    
    if new_mode == optimization_mode and optimizer is not None:
        return 
    
    optimization_mode = new_mode
    optimizer = None  # Reset optimizer when mode changes

    # Update convergence plot title and axis label
    if optimization_mode == "concentration":
        p_conv.title.text = "Optimizer Convergence - Lutein Concentration"
        p_conv.yaxis.axis_label = "Max Lutein Found (g/L)"
        # Hide cost and yield columns
        for col in cost_columns: col.visible = False
        for col in yield_columns: col.visible = False
    elif optimization_mode == "cost": # cost optimization
        p_conv.title.text = "Optimizer Convergence - Revenue Optimization"
        p_conv.yaxis.axis_label = "Best Revenue (Profit - Cost) [$]"
        # Show cost columns, hide yield columns
        for col in cost_columns: col.visible = True
        for col in yield_columns: col.visible = False
    else: # optimization_mode == "yield"
        p_conv.title.text = "Optimizer Convergence - Lutein Yield Optimization"
        p_conv.yaxis.axis_label = "Best Lutein Yield (%)"
        # Hide cost columns, show yield columns
        for col in cost_columns: col.visible = False
        for col in yield_columns: col.visible = True
    
    # Update status message if applicable
    if not experiments_source.data['C_x0']:
        update_status("🟢 Ready. Select objective and define parameters, then generate initial points.")
    else:
        process_and_plot_latest_results() 
        update_status(f"Objective changed to '{optimization_mode}'. Recalculated best results.")
    
    set_ui_state()


def set_ui_state(lock_all=False):
    """Central function to manage the enabled/disabled state of all buttons and inputs."""
    if lock_all:
        for w in all_buttons: w.disabled = True
        for w in param_and_settings_widgets: w.disabled = True
        return

    has_points = len(experiments_source.data['C_x0']) > 0
    has_uncalculated_points = has_points and any(np.isnan(v) for v in experiments_source.data['Lutein'])
    has_calculated_points = has_points and not has_uncalculated_points
    is_preview_pending = previewed_point is not None

    for widget in param_and_settings_widgets: 
        # Only disable parameter/settings widgets if points have been generated
        widget.disabled = has_points 
    
    generate_button.disabled = has_points
    reset_button.disabled = not has_points
    calculate_button.disabled = not has_uncalculated_points
    suggest_button.disabled = not has_calculated_points or is_preview_pending
    run_suggestion_button.disabled = not is_preview_pending

def get_current_dimensions():
    """Reads parameter ranges from UI and creates skopt dimension objects."""
    try:
        return [
            Real(cx0_range.value[0], cx0_range.value[1], name="C_x0"),
            Real(cn0_range.value[0], cn0_range.value[1], name="C_N0"),
            Real(fin_range.value[0], fin_range.value[1], name="F_in", prior='log-uniform'),
            Real(cnin_range.value[0], cnin_range.value[1], name="C_N_in"),
            Real(i0_range.value[0], i0_range.value[1], name="I0"),
        ]
    except Exception as e:
        update_status(f"❌ Error creating dimensions: {e}")
        return None

def reset_experiment():
    """Resets the entire application state to the beginning."""
    global optimization_history, optimizer, previewed_point
    optimization_history.clear()
    optimizer = None
    previewed_point = None

    experiments_source.data = {k: [] for k in experiments_source.data}
    convergence_source.data = {k: [] for k in convergence_source.data}
    simulation_source.data = {k: [] for k in simulation_source.data}
    
    suggestion_div.text = ""
    results_div.text = ""
    update_status("🟢 Ready. Define parameters and generate initial points.")
    set_optimization_mode() # Reapply visibility settings
    set_ui_state()

def generate_initial_points():
    """Generates initial experimental points based on UI settings."""
    doc.add_next_tick_callback(lambda: update_status("🔄 Generating initial points..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))
    
    def worker():
        dims = get_current_dimensions()
        if dims is None: 
            doc.add_next_tick_callback(set_ui_state)
            return

        n_initial = n_initial_input.value
        sampler_choice = sampler_select.value
        
        try:
            if sampler_choice == 'LHS':
                sampler = Lhs(lhs_type="centered", criterion="maximin")
                x0 = sampler.generate(dims, n_samples=n_initial)
            elif sampler_choice == 'Sobol':
                sampler = Sobol()
                x0 = sampler.generate(dims, n_samples=n_initial, random_state=np.random.randint(1000))
            else: # Random
                x0 = [ [d.rvs(1)[0] for d in dims] for _ in range(n_initial) ]

            new_data = {name.name: [point[i] for point in x0] for i, name in enumerate(dims)}
            new_data.update({
                'Lutein': [np.nan] * n_initial,
                'Total_Cost': [np.nan] * n_initial,
                'Lutein_Profit': [np.nan] * n_initial,
                'Revenue_J': [np.nan] * n_initial,
                'Lutein_Yield': [np.nan] * n_initial # Initialize new column
            })
            
            def callback():
                experiments_source.data = new_data
                update_status("🟢 Generated initial points. Ready to calculate.")
                set_ui_state()
            doc.add_next_tick_callback(callback)

        except Exception as e:
            doc.add_next_tick_callback(lambda: update_status(f"❌ Error generating points: {e}"))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()


def calculate_lutein_for_table():
    """Runs simulation for the points in the table."""
    doc.add_next_tick_callback(lambda: update_status("🔄 Calculating Lutein and costs for initial points..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        try:
            # Find indices of uncalculated points
            nan_indices = [i for i, v in enumerate(experiments_source.data['Lutein']) if np.isnan(v)]
            if not nan_indices:
                doc.add_next_tick_callback(lambda: update_status("🟢 All points already calculated."))
                doc.add_next_tick_callback(set_ui_state)
                return
                
            points_to_calc = []
            for i in nan_indices: 
                points_to_calc.append([experiments_source.data[name][i] for name in ['C_x0', 'C_N0', 'F_in', 'C_N_in', 'I0']])

            results = []
            for i, p in enumerate(points_to_calc):
                doc.add_next_tick_callback(lambda i=i: update_status(f"🔄 Calculating point {i+1}/{len(points_to_calc)}..."))
                
                # --- Run simulation to get final C_X and C_L ---
                global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
                temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
                C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = p # Set globals for pbr function
                sol = solve_ivp(pbr, [0, 150], [p[0], p[1], 0.0], t_eval=[150], method="RK45")
                C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 # Restore

                lutein_conc = sol.y[2, -1]
                biomass_conc = sol.y[0, -1] # Get final biomass concentration

                obj_val = _evaluate_lutein_model_objective(*p) # This will now use the current optimization_mode
                
                cost_analysis = calculate_total_cost_and_profit(p, lutein_conc)
                lutein_yield = calculate_lutein_yield(biomass_conc, lutein_conc) # Calculate yield
                
                results.append({
                    'lutein': lutein_conc,
                    'total_cost': cost_analysis['total_cost'],
                    'lutein_profit': cost_analysis['lutein_profit'],
                    'revenue_j': cost_analysis['revenue_J'],
                    'lutein_yield': lutein_yield # Add to results
                })
                
                # Add to optimization history only if not already present (for initial points)
                if not any(np.array_equal(p, item[0]) for item in optimization_history):
                    optimization_history.append([p, obj_val])

            def callback():
                current_data = experiments_source.data.copy() # Make a copy to modify
                for i, res_idx in enumerate(nan_indices):
                    current_data['Lutein'][res_idx] = results[i]['lutein']
                    current_data['Total_Cost'][res_idx] = results[i]['total_cost']
                    current_data['Lutein_Profit'][res_idx] = results[i]['lutein_profit']
                    current_data['Revenue_J'][res_idx] = results[i]['revenue_j']
                    current_data['Lutein_Yield'][res_idx] = results[i]['lutein_yield'] # Update new column
                
                experiments_source.data = current_data # Update ColumnDataSource
                update_status("✅ Calculation complete. Ready to get a suggestion.")
                process_and_plot_latest_results() # Update plots and best results
                set_ui_state()
            doc.add_next_tick_callback(callback)
            
        except Exception as e:
            error_message = f"❌ Error during calculation: {e}"
            doc.add_next_tick_callback(lambda: update_status(error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()

def _ensure_optimizer_is_ready():
    """Internal helper to create and prime the optimizer if it doesn't exist."""
    global optimizer
    if optimizer is None:
        dims = get_current_dimensions()
        x_history = [item[0] for item in optimization_history]
        y_history = [item[1] for item in optimization_history]
        
        optimizer = Optimizer(
            dimensions=dims,
            base_estimator=surrogate_select.value,
            acq_func=acq_func_select.value,
            n_initial_points=len(x_history), 
            random_state=np.random.randint(1000)
        )
        
        if x_history:
            optimizer.tell(x_history, y_history)

def suggest_next_experiment():
    """Asks the optimizer for the next best point to sample, without running it."""
    global previewed_point
    doc.add_next_tick_callback(lambda: update_status("🔄 Getting next suggestion preview..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        global previewed_point
        try:
            _ensure_optimizer_is_ready()
            next_point = optimizer.ask()
            previewed_point = next_point # Store for the 'Run' button
            mean, std = optimizer.models[-1].predict([next_point], return_std=True)
            
            metric_text = ""
            if optimization_mode == "concentration":
                predicted_lutein = -mean[0]
                uncertainty = std[0]
                metric_text = f"<b>Predicted Lutein: {predicted_lutein:.4f} ± {uncertainty:.4f} g/L</b>"
            elif optimization_mode == "cost":
                predicted_revenue = mean[0]
                uncertainty = std[0]
                metric_text = f"<b>Predicted Revenue (J): ${predicted_revenue:.4f} ± {uncertainty:.4f}</b>"
            else: # optimization_mode == "yield"
                predicted_yield = -mean[0] # Convert back to positive yield
                uncertainty = std[0]
                metric_text = f"<b>Predicted Lutein Yield: {predicted_yield:.4f} ± {uncertainty:.4f} %</b>"

            def callback():
                names = [d.name for d in get_current_dimensions()]
                suggestion_html = "<h5>Suggested Next Experiment:</h5>"
                suggestion_html += metric_text + "<ul>"
                for name, val in zip(names, next_point):
                    suggestion_html += f"<li><b>{name}:</b> {val:.4f}</li>"
                suggestion_html += "</ul>"
                suggestion_div.text = suggestion_html
                update_status("💡 Suggestion received. You can now run this specific experiment.")
                set_ui_state()
            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error getting preview: {e}"
            doc.add_next_tick_callback(lambda: update_status(error_message))
            doc.add_next_tick_callback(set_ui_state)
    threading.Thread(target=worker).start()
    
def run_suggestion():
    """Runs the specific experiment that was previewed."""
    if previewed_point is None: return
    doc.add_next_tick_callback(lambda: update_status("🔄 Running suggested experiment..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        global previewed_point
        try:
            point_to_run = previewed_point
            obj_val = _evaluate_lutein_model_objective(*point_to_run)
            
            # Calculate lutein concentration and cost analysis for the simulation plot and table
            global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
            temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
            C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = point_to_run
            sol = solve_ivp(pbr, [0, 150], [point_to_run[0], point_to_run[1], 0.0], t_eval=[150], method="RK45")
            C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 # Restore
            lutein_val = sol.y[2, -1]
            biomass_val = sol.y[0, -1] # Get final biomass
            cost_analysis = calculate_total_cost_and_profit(point_to_run, lutein_val)
            lutein_yield_val = calculate_lutein_yield(biomass_val, lutein_val) # Calculate yield
            
            optimizer.tell(point_to_run, obj_val)
            optimization_history.append([point_to_run, obj_val])
            
            def callback():
                global previewed_point
                new_point_data = {
                    'C_x0': [point_to_run[0]],
                    'C_N0': [point_to_run[1]],
                    'F_in': [point_to_run[2]],
                    'C_N_in': [point_to_run[3]],
                    'I0': [point_to_run[4]],
                    'Lutein': [lutein_val],
                    'Total_Cost': [cost_analysis['total_cost']],
                    'Lutein_Profit': [cost_analysis['lutein_profit']],
                    'Revenue_J': [cost_analysis['revenue_J']],
                    'Lutein_Yield': [lutein_yield_val] # Add to stream
                }
                experiments_source.stream(new_point_data)
                opt_step_number = len(optimization_history) - n_initial_input.value
                
                if optimization_mode == "concentration":
                    update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Lutein: {lutein_val:.4f} g/L")
                elif optimization_mode == "cost":
                    update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Revenue: ${cost_analysis['revenue_J']:.4f}")
                else: # optimization_mode == "yield"
                    update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Lutein Yield: {lutein_yield_val:.4f} %")
                
                previewed_point = None
                suggestion_div.text = ""
                process_and_plot_latest_results()
                set_ui_state()
            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error running suggestion: {e}"
            doc.add_next_tick_callback(lambda: update_status(error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()

def process_and_plot_latest_results():
    """Finds the best result from the history and updates plots."""
    if not optimization_history: 
        results_div.text = "" # Clear results if no history
        return
    
    # Sort history to find the actual best (minimum objective value)
    # The objective value is either -Lutein (maximize Lutein) or J (minimize J) or -Yield (maximize Yield)
    best_item = min(optimization_history, key=lambda item: item[1])
    best_params, best_obj_val = best_item[0], best_item[1]
    
    # Calculate lutein concentration for best parameters (need to set globals for pbr)
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = best_params # Set globals for pbr function
    sol = solve_ivp(pbr, [0, 150], [best_params[0], best_params[1], 0.0], t_eval=[150], method="RK45")
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 # Restore
    
    max_lutein = sol.y[2, -1]
    final_biomass = sol.y[0, -1] # Get final biomass
    cost_analysis = calculate_total_cost_and_profit(best_params, max_lutein)
    best_yield = calculate_lutein_yield(final_biomass, max_lutein) # Calculate yield
    
    optimal_params = {dim.name: val for dim, val in zip(get_current_dimensions(), best_params)}

    results_html = f"<h3>Overall Best Result So Far</h3>"
    if optimization_mode == "concentration":
        results_html += f"<b>Maximum Lutein Found:</b> {max_lutein:.4f} g/L<br/>"
    elif optimization_mode == "cost": # cost optimization
        results_html += f"<b>Best Revenue (J):</b> ${cost_analysis['revenue_J']:.4f}<br/>"
        results_html += f"<b>Lutein Concentration:</b> {max_lutein:.4f} g/L<br/>"
        results_html += f"<b>Total Cost:</b> ${cost_analysis['total_cost']:.4f}<br/>"
        results_html += f"<b>Lutein Profit:</b> ${cost_analysis['lutein_profit']:.4f}<br/>"
    else: # yield optimization
        results_html += f"<b>Maximum Lutein Yield Found:</b> {best_yield:.4f} %<br/>"
        results_html += f"<b>Lutein Concentration:</b> {max_lutein:.4f} g/L<br/>"
        # Optionally, you might want to show cost/profit even for yield optimization
        results_html += f"<b>Total Cost:</b> ${cost_analysis['total_cost']:.4f}<br/>"
        results_html += f"<b>Lutein Profit:</b> ${cost_analysis['lutein_profit']:.4f}<br/>"
    
    results_html += "<b>Corresponding Parameters:</b><ul>"
    for param, value in optimal_params.items(): 
        results_html += f"<li><b>{param}:</b> {value:.4f}</li>"
    results_html += "</ul>"
    results_div.text = results_html
    
    update_convergence_plot_from_history()
    run_final_simulation(best_params)

def update_convergence_plot_from_history():
    """Recalculates and updates the entire convergence plot from the history."""
    num_initial = n_initial_input.value
    opt_history = optimization_history[num_initial:]
    if not opt_history:
        convergence_source.data = dict(iter=[], best_value=[])
        return
        
    iters = list(range(1, len(opt_history) + 1))
    best_values_so_far = []
    
    initial_points_history = optimization_history[:num_initial]
    
    current_best_initial = None
    if initial_points_history:
        if optimization_mode == "concentration" or optimization_mode == "yield":
            # For concentration and yield, we minimized negative value, so initial best is -min(obj_val)
            current_best_initial = -min(p[1] for p in initial_points_history) 
        else: # cost optimization
            # For cost, we minimized J, so initial best is min(obj_val)
            current_best_initial = min(p[1] for p in initial_points_history)

    current_best = current_best_initial if current_best_initial is not None else (-np.inf if optimization_mode in ["concentration", "yield"] else np.inf)

    for _, y_val in opt_history:
        if optimization_mode == "concentration" or optimization_mode == "yield":
            value = -y_val  # Convert back to positive for plotting
            if value > current_best: current_best = value
        else: # cost optimization
            value = y_val
            if value < current_best: current_best = value
        best_values_so_far.append(current_best)
        
    convergence_source.data = {'iter': iters, 'best_value': best_values_so_far}
    p_conv.xaxis.ticker = FixedTicker(ticks=iters)
    if iters:
        p_conv.x_range.end = iters[-1] + 0.5


def run_final_simulation(best_params):
    """Runs and plots a full simulation using the provided parameter set."""
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    # Temporarily set global model parameters for the pbr function
    temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = best_params
    
    t_eval = np.linspace(0, 150, 300)
    initial_conditions = [best_params[0], best_params[1], 0.0]
    sol = solve_ivp(pbr, [0, 150], initial_conditions, t_eval=t_eval, method="RK45")
    
    # Restore original global parameters
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0

    simulation_source.data = {
        "time": sol.t, 
        "C_X": np.maximum(0, sol.y[0]), 
        "C_N": np.maximum(0, sol.y[1]), 
        "C_L": np.maximum(0, sol.y[2]), 
        "C_L_scaled": np.maximum(0, sol.y[2]) * 100
    }

def update_status(message): 
    status_div.text = message

# --- UI Widgets ---
title_div = Div(text="<h1>Lutein Production Bayesian Optimizer with Cost Analysis</h1>")
description_p = Paragraph(text="""This application uses Bayesian Optimization to find optimal operating conditions for a photobioreactor. You can optimize for maximum lutein concentration or minimum cost. Follow the steps to run a virtual experiment.""", width=450)

# Optimization Objective Selection
objective_title = Div(text="<h4>0. Select Optimization Objective</h4>")
objective_select = Select(title="Optimization Objective:", value="concentration", 
                          options=[("concentration", "Maximize Lutein Concentration"), 
                                   ("cost", "Minimize Cost (Maximize Revenue)"),
                                   ("yield", "Maximize Lutein Yield")])
objective_select.on_change('value', lambda attr, old, new: set_optimization_mode())

# Step 1: Parameter Ranges
param_range_title = Div(text="<h4>1. Define Parameter Search Space</h4>")
cx0_range = RangeSlider(title="C_x0 Range (g/L)", start=0, end=10, value=(0.2, 2.0), step=0.1)
cn0_range = RangeSlider(title="C_N0 Range (g/L)", start=0, end=10, value=(0.2, 2.0), step=0.1)
fin_range = RangeSlider(title="F_in Range", start=1e-5, end=1.5e-1, value=(1e-3, 1.5e-2), step=1e-4, format="0.0000")
cnin_range = RangeSlider(title="C_N_in Range (g/L)", start=0, end=50, value=(5.0, 15.0), step=0.5)
i0_range = RangeSlider(title="I0 Range (umol/m2-s)", start=0, end=1000, value=(100, 200), step=10)

# Step 2: Sampler Settings
settings_title = Div(text="<h4>2. Configure Initial Sampling & Model</h4>")
surrogate_select = Select(title="Surrogate Model:", value="GP", options=["GP", "RF", "ET"])
acq_func_select = Select(title="Acquisition Function:", value="gp_hedge", options=["gp_hedge", "EI", "PI", "LCB"])
sampler_select = Select(title="Sampling Method:", value="LHS", options=["LHS", "Sobol", "Random"])
n_initial_input = Spinner(title="Number of Initial Points:", low=1, step=1, value=10, width=150)
param_and_settings_widgets = [objective_select, cx0_range, cn0_range, fin_range, cnin_range, i0_range, surrogate_select, acq_func_select, sampler_select, n_initial_input]

# Step 3: Experiment Workflow
actions_title = Div(text="<h4>3. Run Experiment Workflow</h4>")
generate_button = Button(label="A) Generate Initial Points", button_type="primary", width=400)
calculate_button = Button(label="B) Calculate Lutein & Costs for Initial Points", button_type="default", width=400)
suggest_button = Button(label="C) Suggest Next Experiment & Show Prediction", button_type="success", width=400)
suggestion_div = Div(text="", width=400)
run_suggestion_button = Button(label="D) Run Suggested Experiment & Update Model", button_type="warning", width=400)
reset_button = Button(label="Reset Experiment", button_type="danger", width=400)
all_buttons = [generate_button, calculate_button, suggest_button, run_suggestion_button, reset_button]

generate_button.on_click(generate_initial_points)
calculate_button.on_click(calculate_lutein_for_table)
suggest_button.on_click(suggest_next_experiment)
run_suggestion_button.on_click(run_suggestion)
reset_button.on_click(reset_experiment)

status_div = Div(text="🟢 Ready. Select objective and define parameters, then generate initial points.")
results_div = Div(text="")

# --- Data Table & Plots ---
# Define all columns, including cost-related ones
columns = [
    TableColumn(field="C_x0", title="C_x0", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="C_N0", title="C_N0", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="F_in", title="F_in", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="C_N_in", title="C_N_in", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="I0", title="I0", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Lutein", title="Lutein (g/L)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Total_Cost", title="Total Cost ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Lutein_Profit", title="Lutein Profit ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Revenue_J", title="Revenue J ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Lutein_Yield", title="Lutein Yield (%)", formatter=NumberFormatter(format="0.0000")) # New column for yield
]

# Store references to cost-related columns for easy toggling
cost_columns = [col for col in columns if col.field in ['Total_Cost', 'Lutein_Profit', 'Revenue_J']]
yield_columns = [col for col in columns if col.field in ['Lutein_Yield']] # New list for yield column

data_table = DataTable(source=experiments_source, columns=columns, width=1000, height=280, editable=False)


p_conv = figure(height=300, width=800, title="Optimizer Convergence", x_axis_label="Optimization Step", y_axis_label="Best Value", y_range=DataRange1d(start=0, range_padding=0.1, range_padding_units='percent'))
p_conv.xaxis.formatter = NumeralTickFormatter(format="0")
p_conv.line(x="iter", y="best_value", source=convergence_source, line_width=2)

p_sim = figure(height=300, width=800, title="Simulation with Best Parameters", x_axis_label="Time (hours)", y_axis_label="Biomass & Nitrate Conc. (g/L)", y_range=DataRange1d(start=0))
p_sim.extra_y_ranges = {"lutein_range": DataRange1d(start=0)}
p_sim.add_layout(LinearAxis(y_range_name="lutein_range", axis_label="Lutein Conc. (x100) [g/L]"), 'right')

p_sim.line(x="time", y="C_X", source=simulation_source, color="green", line_width=2, legend_label="Biomass (C_X)")
p_sim.line(x="time", y="C_N", source=simulation_source, color="blue", line_width=2, legend_label="Nitrate (C_N)")
p_sim.line(x="time", y="C_L_scaled", source=simulation_source, color="orange", line_width=3, legend_label="Lutein (C_L) x100", y_range_name="lutein_range")
p_sim.legend.location = "top_left"
p_sim.legend.click_policy = "hide"

# --- Layout ---
controls_col = column(
    title_div, description_p,
    objective_title, objective_select,
    param_range_title, cx0_range, cn0_range, fin_range, cnin_range, i0_range,
    settings_title, surrogate_select, acq_func_select, sampler_select, n_initial_input,
    actions_title, generate_button, calculate_button, suggest_button, suggestion_div, run_suggestion_button, reset_button,
    status_div,
    width=470,
)
results_col = column(data_table, results_div, p_conv, p_sim)
layout = row(controls_col, results_col)
doc.add_root(layout)

# Initialize UI
set_optimization_mode()  # Set initial mode (which will also set column visibility)
set_ui_state()



#for the cost equations, use the following codes and thought process. 
# cost