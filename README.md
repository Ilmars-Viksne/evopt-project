# evopt - Enterprise-grade Evolutionary Algorithm Optimizer

`evopt` is a robust, modular, and extensible framework for solving complex optimization problems using Evolutionary Algorithms (EAs). Built with a focus on clean architecture and dependency injection, it allows developers to easily swap out optimization strategies for initialization, selection, crossover, mutation, and fitness evaluation.

## Features

- **Modular Architecture**: Uses the Strategy Pattern to keep the core evolution engine independent of specific implementations.
- **Production-Ready**: Built with modern Python (>=3.10), leveraging `dataclasses` for domain entities and `Pydantic` for validated configuration.
- **Extensible**: Easily implement custom fitness functions, selection methods, or genetic operators by subclassing provided base interfaces.
- **Built-in Implementations**: Includes standard strategies like Tournament Selection, Gaussian Mutation, and Average Crossover.
- **CLI Support**: Comes with a built-in CLI for running default mathematical optimizations.
- **Enterprise Utilities**: Integrated logging and exception handling for reliable execution in production environments.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd evopt-project
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Project Structure

```text
evopt-project/
├── pyproject.toml           # Project configuration and dependencies
├── rocket_optimizer.py      # Example consumer script: Rocket Thrust Optimization
├── src/
│   └── evopt/
│       ├── cli.py           # Command-line interface entry point
│       ├── config.py        # Pydantic-based configuration management
│       ├── core/            # Domain logic and engine
│       │   ├── engine.py    # The core evolutionary orchestration engine
│       │   ├── entities.py  # Data models (Individual, EvolutionResult)
│       │   └── exceptions.py# Package-specific exceptions
│       ├── implementations/ # Concrete strategy implementations
│       │   ├── crossover.py
│       │   ├── fitness.py
│       │   ├── initialization.py
│       │   ├── mutation.py
│       │   └── selection.py
│       └── interfaces/      # Abstract base classes for EA strategies
│           └── strategies.py
└── tests/                   # Unit tests
    └── test_engine.py
```

## Usage

### Using the CLI

Run the default 2D mathematical optimization:
```bash
evopt-cli
```
*Note: This requires the package to be installed.*

### Running the Rocket Optimizer Example

`rocket_optimizer.py` demonstrates how to use `evopt` to solve a complex, multi-dimensional physics problem (optimizing a rocket's thrust profile over time):

```bash
python rocket_optimizer.py
```

### Implementing Custom Strategies

To solve your own optimization problem, implement the `FitnessStrategy` interface:

```python
from evopt.interfaces.strategies import FitnessStrategy

class MyCustomFitness(FitnessStrategy):
    def evaluate(self, genes: list[float]) -> float:
        # Calculate fitness based on genes
        return sum(genes)
```

Then, inject it into the `EvolutionEngine`:

```python
from evopt.core.engine import EvolutionEngine
# ... import other strategies ...

engine = EvolutionEngine(
    initialization_strategy=init_strategy,
    fitness_strategy=MyCustomFitness(),
    selection_strategy=selection_strategy,
    crossover_strategy=crossover_strategy,
    mutation_strategy=mutation_strategy
)

result = engine.run()
print(f"Best solution: {result.best_individual.genes}")
```

## Configuration

`evopt` uses `pydantic-settings` to manage configuration. You can override defaults using environment variables or a `.env` file. Refer to `src/evopt/config.py` for available settings (e.g., `POPULATION_SIZE`, `GENERATIONS`, `MUTATION_RATE`).

## Testing

Run the test suite using `pytest`:
```bash
pytest
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.
