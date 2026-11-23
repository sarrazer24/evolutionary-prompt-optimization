
import random

from src.genetic_algorithm.population import Individual


def tournament_selection(
    population: list[Individual],
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> Individual:
    """Select one parent using tournament selection."""

    if not population:
        raise ValueError("Population cannot be empty.")

    if tournament_size < 1:
        raise ValueError("Tournament size must be at least 1.")

    if any(ind.fitness is None for ind in population):
        raise ValueError("All individuals must be evaluated before selection.")

    if rng is None:
        rng = random.Random()

    contestants = rng.sample(
        population,
        k=min(tournament_size, len(population)),
    )

    winner = max(contestants, key=lambda ind: ind.fitness)

    return winner


def select_elites(
    population: list[Individual],
    elite_count: int = 2,
) -> list[Individual]:
    """Return the top individuals ranked by fitness."""

    if elite_count < 0:
        raise ValueError("Elite count cannot be negative.")

    if elite_count > len(population):
        raise ValueError("Elite count cannot exceed population size.")

    if any(ind.fitness is None for ind in population):
        raise ValueError("All individuals must be evaluated before elitism.")

    ranked_population = sorted(
        population,
        key=lambda ind: ind.fitness,
        reverse=True,
    )

    return ranked_population[:elite_count].copy()
