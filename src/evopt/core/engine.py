import logging
import random
from evopt.core.entities import Individual, EvolutionResult
from evopt.core.exceptions import ConfigurationError
from evopt.interfaces.strategies import (
    InitializationStrategy,
    FitnessStrategy,
    SelectionStrategy,
    CrossoverStrategy,
    MutationStrategy
)

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """
    Core Evolutionary Algorithm engine.

    Notes:
        This class uses Constructor Dependency Injection. It does not know *how*
        individuals are initialized, evaluated, or modified. It only orchestrates
        the evolutionary loop using the provided strategies.
    """
    def __init__(
        self,
        initialization_strategy: InitializationStrategy,
        fitness_strategy: FitnessStrategy,
        selection_strategy: SelectionStrategy,
        crossover_strategy: CrossoverStrategy,
        mutation_strategy: MutationStrategy,
        population_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        generations: int = 100
    ):
        if population_size < 2:
            raise ConfigurationError("Population size must be at least 2.")

        self.init_strategy = initialization_strategy
        self.fitness_strategy = fitness_strategy
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy

        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations

    def _evaluate_population(self, population: list[Individual]) -> None:
        """Evaluates individuals that lack a fitness score."""
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self.fitness_strategy.evaluate(ind.genes)

    def run(self) -> EvolutionResult:
        """
        Executes the evolutionary loop.

        Returns:
            EvolutionResult: The outcome of the evolution process.
        """
        logger.info(f"Starting evolution for {self.generations} generations.")

        population = self.init_strategy.initialize(self.population_size)
        self._evaluate_population(population)

        best_overall = max(population, key=lambda ind: ind.fitness) # type: ignore

        for gen in range(self.generations):
            # Elitism: carry over the best individual
            current_best = max(population, key=lambda ind: ind.fitness) # type: ignore
            if current_best.fitness > best_overall.fitness: # type: ignore
                best_overall = current_best

            next_generation: list[Individual] = [current_best]

            while len(next_generation) < self.population_size:
                parent1 = self.selection_strategy.select(population)
                parent2 = self.selection_strategy.select(population)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover_strategy.crossover(parent1, parent2)
                else:
                    # Deep copy genes to prevent reference mutation
                    child = Individual(genes=parent1.genes.copy())

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutation_strategy.mutate(child)

                next_generation.append(child)

            population = next_generation
            self._evaluate_population(population)

            if (gen + 1) % 10 == 0:
                logger.info(f"Generation {gen + 1}/{self.generations} completed. Best fitness: {best_overall.fitness:.6f}")

        logger.info("Evolution finished.")
        return EvolutionResult(
            best_individual=best_overall,
            best_fitness=best_overall.fitness, # type: ignore
            generations_completed=self.generations
        )
