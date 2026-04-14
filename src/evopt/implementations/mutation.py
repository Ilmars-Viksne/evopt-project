import random
from evopt.interfaces.strategies import MutationStrategy
from evopt.core.entities import Individual

class GaussianMutationStrategy(MutationStrategy):
    """Applies Gaussian noise to genes, bounded by search space limits."""

    def __init__(self, search_min: float, search_max: float, mu: float = 0.0, sigma: float = 0.3):
        self.search_min = search_min
        self.search_max = search_max
        self.mu = mu
        self.sigma = sigma

    def mutate(self, individual: Individual) -> Individual:
        mutated_genes =[]
        for gene in individual.genes:
            mutated_gene = gene + random.gauss(self.mu, self.sigma)
            mutated_gene = max(self.search_min, min(self.search_max, mutated_gene))
            mutated_genes.append(mutated_gene)

        return Individual(genes=mutated_genes)
