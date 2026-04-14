import pytest
from evopt.core.engine import EvolutionEngine
from evopt.core.entities import Individual
from evopt.interfaces.strategies import (
    InitializationStrategy, FitnessStrategy, SelectionStrategy,
    CrossoverStrategy, MutationStrategy
)

class MockInitStrategy(InitializationStrategy):
    def initialize(self, population_size: int) -> list[Individual]:
        return [Individual(genes=[1.0, 1.0]) for _ in range(population_size)]

class MockFitnessStrategy(FitnessStrategy):
    def evaluate(self, genes: list[float]) -> float:
        return sum(genes)

class MockSelectionStrategy(SelectionStrategy):
    def select(self, population: list[Individual]) -> Individual:
        return population[0]

class MockCrossoverStrategy(CrossoverStrategy):
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        return Individual(genes=[p1 + p2 for p1, p2 in zip(parent1.genes, parent2.genes)])

class MockMutationStrategy(MutationStrategy):
    def mutate(self, individual: Individual) -> Individual:
        return Individual(genes=[g + 0.1 for g in individual.genes])

def test_engine_initialization_and_evaluation():
    engine = EvolutionEngine(
        initialization_strategy=MockInitStrategy(),
        fitness_strategy=MockFitnessStrategy(),
        selection_strategy=MockSelectionStrategy(),
        crossover_strategy=MockCrossoverStrategy(),
        mutation_strategy=MockMutationStrategy(),
        population_size=10,
        generations=1
    )
    result = engine.run()
    assert result.generations_completed == 1
    assert result.best_individual is not None
    assert result.best_fitness >= 2.0

def test_engine_dependency_injection_isolation():
    engine = EvolutionEngine(
        initialization_strategy=MockInitStrategy(),
        fitness_strategy=MockFitnessStrategy(),
        selection_strategy=MockSelectionStrategy(),
        crossover_strategy=MockCrossoverStrategy(),
        mutation_strategy=MockMutationStrategy(),
        population_size=5,
        generations=5,
        crossover_rate=1.0,
        mutation_rate=0.0
    )
    result = engine.run()
    assert result.best_fitness > 2.0
    assert len(result.best_individual.genes) == 2
