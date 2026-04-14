import random
from evopt.interfaces.strategies import InitializationStrategy
from evopt.core.entities import Individual

class RandomInitializationStrategy(InitializationStrategy):
    """Initializes population with random uniform values within bounds."""

    def __init__(self, search_min: float, search_max: float, dimensions: int = 2):
        self.search_min = search_min
        self.search_max = search_max
        self.dimensions = dimensions

    def initialize(self, population_size: int) -> list[Individual]:
        population =[]
        for _ in range(population_size):
            genes =[random.uniform(self.search_min, self.search_max) for _ in range(self.dimensions)]
            population.append(Individual(genes=genes))
        return population
