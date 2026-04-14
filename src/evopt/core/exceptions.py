class EvolutionOptimizerError(Exception):
    """Base exception for the evopt package."""
    pass

class ConfigurationError(EvolutionOptimizerError):
    """Raised when there is a configuration error."""
    pass

class EvaluationError(EvolutionOptimizerError):
    """Raised when an individual cannot be evaluated."""
    pass
