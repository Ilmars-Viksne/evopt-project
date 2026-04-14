from evopt.interfaces.strategies import CrossoverStrategy
from evopt.core.entities import Individual

class AverageCrossoverStrategy(CrossoverStrategy):
    """Creates a child by averaging the genes of two parents."""

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child_genes =[
            (g1 + g2) / 2.0
            for g1, g2 in zip(parent1.genes, parent2.genes)
        ]
        return Individual(genes=child_genes)
