from evopt.interfaces.strategies import FitnessStrategy

class PenalizedFitnessStrategy(FitnessStrategy):
    """
    Standard fitness strategy applying penalties for constraint violations.
    Used for the original 2D math problem prototype.
    """
    def __init__(self, penalty_factor: float = 1000.0):
        self.penalty_factor = penalty_factor

    def _objective_function(self, x1: float, x2: float) -> float:
        return (x1 - 1)**2 + x2**2

    def _constraint_g1(self, x1: float, x2: float) -> float:
        return 2 * x1**2 - x2 - 2

    def _constraint_g2(self, x1: float, x2: float) -> float:
        return x1 + 2 * x2

    def evaluate(self, genes: list[float]) -> float:
        x1, x2 = genes[0], genes[1]

        f_value = self._objective_function(x1, x2)
        g1_violation = max(0.0, self._constraint_g1(x1, x2))
        g2_violation = max(0.0, self._constraint_g2(x1, x2))

        total_penalty = self.penalty_factor * (g1_violation + g2_violation)
        penalized_f = f_value + total_penalty

        return -penalized_f
