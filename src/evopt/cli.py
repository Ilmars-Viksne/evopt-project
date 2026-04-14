import logging
from evopt.config import EvolutionConfig
from evopt.core.engine import EvolutionEngine
from evopt.implementations.fitness import PenalizedFitnessStrategy
from evopt.implementations.initialization import RandomInitializationStrategy
from evopt.implementations.crossover import AverageCrossoverStrategy
from evopt.implementations.mutation import GaussianMutationStrategy
from evopt.implementations.selection import TournamentSelectionStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    """CLI Entry Point for the default 2D mathematical optimization."""
    logger.info("Initializing evopt...")

    config = EvolutionConfig()

    init_strategy = RandomInitializationStrategy(
        search_min=config.search_space_min, search_max=config.search_space_max
    )
    fitness_strategy = PenalizedFitnessStrategy(penalty_factor=config.penalty_factor)
    selection_strategy = TournamentSelectionStrategy(tournament_size=config.tournament_size)
    crossover_strategy = AverageCrossoverStrategy()
    mutation_strategy = GaussianMutationStrategy(
        search_min=config.search_space_min, search_max=config.search_space_max, sigma=config.mutation_sigma
    )

    engine = EvolutionEngine(
        initialization_strategy=init_strategy,
        fitness_strategy=fitness_strategy,
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        mutation_strategy=mutation_strategy,
        population_size=config.population_size,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        generations=config.generations
    )

    result = engine.run()

    best_genes = result.best_individual.genes
    min_value = -result.best_fitness

    print("\n" + "="*50)
    print("EVOLUTIONARY OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Optimal Solution Found: x1 = {best_genes[0]:.6f}, x2 = {best_genes[1]:.6f}")
    print(f"Minimum Function Value: f(x1, x2) = {min_value:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
