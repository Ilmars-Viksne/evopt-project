from pydantic_settings import BaseSettings, SettingsConfigDict

class EvolutionConfig(BaseSettings):
    """
    Application configuration validated by Pydantic.
    Reads from environment variables or a .env file.
    """
    population_size: int = 100
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    tournament_size: int = 5

    search_space_min: float = -3.0
    search_space_max: float = 3.0
    penalty_factor: float = 1000.0
    mutation_sigma: float = 0.3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
