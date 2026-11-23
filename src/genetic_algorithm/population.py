import random
from dataclasses import dataclass, field
from typing import Any

from src.prompts.initial_population import generate_initial_prompts

@dataclass
class Individual:
    """Represents one candidate prompt in the genetic algorithm."""

    id: int
    prompt: str
    generation: int

    parent_ids: tuple[int, ...] = ()
    fitness: float | None = None
    accuracy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)



def initialize_population(
    llm,
    population_size: int = 12,
    seed_prompts: list[str] | None = None,
    rng: random.Random | None = None,
    start_id: int = 0,
) -> list[Individual]:
    """Generate and initialize the first population."""

    if rng is None:
        rng = random.Random()

    prompts = generate_initial_prompts(
        llm,
        n=population_size,
        seed_prompts=seed_prompts,
        rng=rng,
    )

    population = []

    for i, prompt in enumerate(prompts):
        individual = Individual(
            id=start_id + i,
            prompt=prompt,
            generation=0,
        )
        population.append(individual)

    return population
