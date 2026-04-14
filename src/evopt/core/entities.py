from dataclasses import dataclass, field

@dataclass
class Individual:
    """
    Represents a single solution in the population.

    Notes:
        Implemented as a pure Python dataclass to keep the core domain
        free of external framework dependencies (like Pydantic).
    """
    genes: list[float]
    fitness: float | None = None

@dataclass
class EvolutionResult:
    """
    Encapsulates the final result of the evolutionary process.
    """
    best_individual: Individual
    best_fitness: float
    generations_completed: int
    population_history: list[list[Individual]] = field(default_factory=list)
