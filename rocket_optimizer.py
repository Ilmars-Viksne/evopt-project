import numpy as np
import logging

# Notice how clean the imports are now using 'evopt'
from evopt.core.engine import EvolutionEngine
from evopt.interfaces.strategies import FitnessStrategy
from evopt.implementations.initialization import RandomInitializationStrategy
from evopt.implementations.crossover import AverageCrossoverStrategy
from evopt.implementations.mutation import GaussianMutationStrategy
from evopt.implementations.selection import TournamentSelectionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Problem Constants ---
g = 9.81
k_d = 0.04
k_m = 0.001
F_max = 200.0
dt = 2.0
m_0 = 10.0
fuel_mass = 8.0
N_STEPS = 13
PENALTY_FACTOR = 10000.0

# --- Rocket Simulation ---
def simulate_rocket_flight(X):
    X = np.array(X)
    n_steps = len(X)
    a = np.zeros(n_steps)
    m = np.zeros(n_steps)
    v = np.zeros(n_steps)
    ds = np.zeros(n_steps)
    s = np.zeros(n_steps)

    m[0] = m_0
    v[0] = 0.0
    a[0] = X[0] / m[0] - g
    ds[0] = 0.5 * a[0] * dt ** 2
    s[0] = ds[0]

    fuel_remaining = fuel_mass

    for i in range(n_steps - 1):
        v[i + 1] = v[i] + a[i] * dt
        fuel_consumed = k_m * abs(X[i]) * dt

        actual_fuel_consumed = min(fuel_consumed, fuel_remaining)
        fuel_remaining -= actual_fuel_consumed

        m[i + 1] = m[i] - actual_fuel_consumed

        current_thrust = X[i + 1] if fuel_remaining > 0 else 0.0

        a[i + 1] = current_thrust / m[i + 1] - g - k_d * v[i + 1] / m[i + 1]
        ds[i + 1] = v[i + 1] * dt + 0.5 * a[i + 1] * dt ** 2
        s[i + 1] = s[i] + ds[i + 1]

    return m, v, a, ds, s

# --- Custom Fitness Strategy ---
class RocketFitnessStrategy(FitnessStrategy):
    def evaluate(self, genes: list[float]) -> float:
        thrust_profile = np.clip(genes, 0.0, F_max)
        m, v, a, ds, s = simulate_rocket_flight(thrust_profile)

        final_s = s[-1]
        final_v = v[-1]
        final_a = a[-1]

        fuel_consumed = m_0 - m[-1]

        pos_violation = max(0.0, 500.0 - final_s) + max(0.0, final_s - 550.0)
        vel_violation = max(0.0, -1.0 - final_v) + max(0.0, final_v - 1.0)
        acc_violation = max(0.0, -0.5 - final_a) + max(0.0, final_a - 0.5)

        total_penalty = PENALTY_FACTOR * (pos_violation + vel_violation + acc_violation)

        return -(fuel_consumed + total_penalty)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Setting up Rocket Thrust Optimization...")

    init_strategy = RandomInitializationStrategy(search_min=0.0, search_max=F_max, dimensions=N_STEPS)
    fitness_strategy = RocketFitnessStrategy()
    selection_strategy = TournamentSelectionStrategy(tournament_size=5)
    crossover_strategy = AverageCrossoverStrategy()
    mutation_strategy = GaussianMutationStrategy(search_min=0.0, search_max=F_max, mu=0.0, sigma=20.0)

    engine = EvolutionEngine(
        initialization_strategy=init_strategy,
        fitness_strategy=fitness_strategy,
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        mutation_strategy=mutation_strategy,
        population_size=500,
        generations=300,
        crossover_rate=0.85,
        mutation_rate=0.4
    )

    result = engine.run()

    best_thrust_profile = np.clip(result.best_individual.genes, 0.0, F_max)
    m, v, a, ds, s = simulate_rocket_flight(best_thrust_profile)

    print("\n" + "="*50)
    print("🚀 ROCKET OPTIMIZATION RESULTS")
    print("="*50)
    print("Optimal Thrust Profile (N) per time step:")
    for i, thrust in enumerate(best_thrust_profile):
        print(f"  t={i*2:02d}s : {thrust:>6.2f} N")

    print("-" * 50)
    print("Final Flight Metrics:")
    print(f"  Final Position     : {s[-1]:>8.2f} m   (Target: 500 to 550)")
    print(f"  Final Velocity     : {v[-1]:>8.2f} m/s (Target: -1.0 to 1.0)")
    print(f"  Final Acceleration : {a[-1]:>8.2f} m/s²(Target: -0.5 to 0.5)")
    print(f"  Fuel Consumed      : {m_0 - m[-1]:>8.2f} kg  (Available: 8.0)")
    print("="*50)
