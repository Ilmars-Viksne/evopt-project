from abc import ABC, abstractmethod
from evopt.core.entities import Individual

class InitializationStrategy(ABC):
    """Strategy for creating the initial population."""
    @abstractmethod
    def initialize(self, population_size: int) -> list[Individual]:
        pass

class FitnessStrategy(ABC):
    """Strategy for evaluating the fitness of an individual."""
    @abstractmethod
    def evaluate(self, genes: list[float]) -> float:
        pass

class SelectionStrategy(ABC):
    """Strategy for selecting an individual from a population."""
    @abstractmethod
    def select(self, population: list[Individual]) -> Individual:
        pass

class CrossoverStrategy(ABC):
    """Strategy for combining two parents to create a child."""
    @abstractmethod
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        pass

class MutationStrategy(ABC):
    """Strategy for mutating an individual."""
    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        pass
