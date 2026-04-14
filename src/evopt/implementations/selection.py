import random
from evopt.interfaces.strategies import SelectionStrategy
from evopt.core.entities import Individual
from evopt.core.exceptions import EvolutionOptimizerError

class TournamentSelectionStrategy(SelectionStrategy):
    """Selects the best individual from a random subset of the population."""

    def __init__(self, tournament_size: int = 5):
        self.tournament_size = tournament_size

    def select(self, population: list[Individual]) -> Individual:
        if not population:
            raise EvolutionOptimizerError("Cannot select from an empty population.")

        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness) # type: ignore
