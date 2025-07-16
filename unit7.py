import numpy as np
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.sampler import Lhs, Sobol
import math

# --- Constants ---
EST_SEC_PER_ITER = 2.5
WARN_THRESHOLD_ITERATIONS = 100
MAX_RECOMMENDED_ITERATIONS = 300

# --- Model Constants (unchanged) ---
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

# --- Globals for pbr inputs ---
C_x0_model = 0.5
C_N0_model = 1.0
F_in_model = 8e-3
C_N_in_model = 10.0
I0_model = 150.0

# --- ODE Model ---
def pbr(t, C):
    C_X, C_N, C_L = C
    if C_X < 1e-9: C_X = 1e-9
    if C_N < 1e-9: C_N = 1e-9
    if C_L < 1e-9: C_L = 1e-9

    I = 2 * I0_model * np.exp(-(TAU * 0.01 * 1000 * C_X))
    Iscaling_u = I / (I + K_S + I**2 / K_I)
    Iscaling_k = I / (I + K_SL + I**2 / K_IL)
    u0 = U_M * Iscaling_u
    k0 = K_M * Iscaling_k

    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X
    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in_model * C_N_in_model
    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X

    return np.array([dCxdt, dCndt, dCldt])

# --- Objective Evaluation (returns negative lutein in mg/L) ---
def _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0):
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model = C_x0
    C_N0_model = C_N0
    F_in_model = F_in
    C_N_in_model = C_N_in
    I0_model = I0

    sol = solve_ivp(pbr, [0, 150], [C_x0_model, C_N0_model, 0.0], t_eval=[150])
    final_lutein_conc_mgL = sol.y[2, -1] * 1000  # convert g/L to mg/L

    if final_lutein_conc_mgL <= 0 or not np.isfinite(final_lutein_conc_mgL):
        return 1e6

    return -final_lutein_conc_mgL  # negate for maximization

# --- Objective Wrapper for skopt ---
@use_named_args([
    Real(0.2, 2.0, name='C_x0'),
    Real(0.2, 2.0, name='C_N0'),
    Real(1e-3, 1.5e-2, name='F_in'),
    Real(5.0, 15.0, name='C_N_in'),
    Real(100.0, 200.0, name='I0')
])
def objective_function(C_x0, C_N0, F_in, C_N_in, I0):
    return _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0)

# --- Optimization Space ---
dimensions = [
    Real(0.2, 2.0, name='C_x0'),
    Real(0.2, 2.0, name='C_N0'),
    Real(1e-3, 1.5e-2, name='F_in'),
    Real(5.0, 15.0, name='C_N_in'),
    Real(100.0, 200.0, name='I0')
]

# --- CLI Input Helpers ---
def get_user_choice(prompt, options):
    choice = -1
    while choice not in range(1, len(options) + 1):
        print(prompt)
        for i, option in enumerate(options):
            print(f"{i+1}. {option['label']}")
        try:
            choice = int(input("Enter the number of your choice: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
    return options[choice - 1]['value']

def get_integer_input(prompt, min_val=1, max_val=None, default_val=None):
    while True:
        full_prompt = prompt
        if default_val is not None:
            full_prompt += f" (default: {default_val})"
        value = input(full_prompt + ": ")
        if value == '' and default_val is not None:
            return default_val
        try:
            value = int(value)
            if value < min_val:
                print(f"Value must be at least {min_val}.")
            elif max_val is not None and value > max_val:
                print(f"Value cannot exceed {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

# --- Main CLI Execution ---
if __name__ == '__main__':
    print("\n--- Lutein Production Bayesian Optimization (CLI) ---")

    surrogate_model_choice = get_user_choice("\nChoose Surrogate Model:", [
        {'value': 'GP', 'label': 'Gaussian Process'},
        {'value': 'RF', 'label': 'Random Forest'},
        {'value': 'ET', 'label': 'Extra Trees'}
    ])

    acquisition_function_choice = get_user_choice("\nChoose Acquisition Function:", [
        {'value': 'gp_hedge', 'label': 'GP Hedge'},
        {'value': 'EI', 'label': 'Expected Improvement'},
        {'value': 'PI', 'label': 'Probability of Improvement'},
        {'value': 'LCB', 'label': 'Lower Confidence Bound'}
    ])

    num_iterations = get_integer_input("Enter total number of optimization iterations", 1, MAX_RECOMMENDED_ITERATIONS, 50)

    initial_sampler_choice = get_user_choice("\nChoose Initial Sampling Method:", [
        {'value': 'random', 'label': 'Random Sampling'},
        {'value': 'lhs', 'label': 'Latin Hypercube Sampling'},
        {'value': 'sobol', 'label': 'Sobol Sequence'}
    ])

    max_initial_points_allowed = max(1, num_iterations - 1)
    default_initial_points = min(10, max_initial_points_allowed)
    num_initial_points = get_integer_input("Enter number of initial points", 1, max_initial_points_allowed, default_initial_points)

    if num_iterations > WARN_THRESHOLD_ITERATIONS:
        print(f"\nWARNING: Estimated time ~{math.ceil(num_iterations * EST_SEC_PER_ITER / 60)} minutes.")
        input("Press Enter to continue...")

    initial_points_x, initial_points_y = [], []
    if num_initial_points > 0:
        print(f"\nGenerating {num_initial_points} initial points using {initial_sampler_choice} sampling...")
        np.random.seed(42)
        if initial_sampler_choice == 'random':
            for _ in range(num_initial_points):
                point = [dim.rvs()[0] for dim in dimensions]
                initial_points_x.append(point)
        elif initial_sampler_choice == 'lhs':
            initial_points_x = Lhs().generate(dimensions, num_initial_points, random_state=42)
        elif initial_sampler_choice == 'sobol':
            initial_points_x = Sobol().generate(dimensions, num_initial_points, random_state=42)

        for i, p in enumerate(initial_points_x):
            print(f"  Evaluating initial point {i+1}/{num_initial_points}...")
            initial_points_y.append(_evaluate_lutein_model_objective(*p))

    print("\nOptimization starting...")
    history = []

    def callback(res):
        best_mgL = -res.fun
        history.append({
            'iteration': len(res.func_vals),
            'objective_value': -res.func_vals[-1],
            'best_so_far': best_mgL
        })

    try:
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            base_estimator=surrogate_model_choice,
            acq_func=acquisition_function_choice,
            n_calls=num_iterations,
            x0=initial_points_x if initial_points_x else None,
            y0=initial_points_y if initial_points_y else None,
            n_initial_points=0 if initial_points_x else 10,
            random_state=42,
            callback=[callback]
        )

        print("\n--- Optimization Complete ---")
        print(f"Maximum Lutein concentration found: {-result.fun:.2f} mg/L")
        print("\nOptimal Parameters:")
        for name, val in zip([d.name for d in dimensions], result.x):
            print(f"  {name}: {val:.4f}")

    except Exception as e:
        print(f"\nError during optimization: {e}")
