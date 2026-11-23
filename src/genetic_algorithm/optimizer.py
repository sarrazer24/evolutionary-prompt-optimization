
import copy
import random

from src.genetic_algorithm.population import (
    Individual,
    initialize_population,
)
from src.genetic_algorithm.selection import (
    tournament_selection,
    select_elites,
)
from src.prompts.operators import (
    mutate_prompt,
    crossover_prompts,
)
from src.evaluation.evaluation import (
    PredictionCache,
    evaluate_prompt,
)
from src.genetic_algorithm.logging import save_optimization_results


def evaluate_population(
    population: list[Individual],
    examples,
    labels,
    llm,
    cache: PredictionCache | None = None,
    batch_size: int | None = None,
    split_name: str = "opt",
) -> list[Individual]:
    """Evaluate every individual and store its fitness and metrics.

    Macro F1 is used as the fitness function.
    """

    for individual in population:
        result = evaluate_prompt(
            prompt=individual.prompt,
            examples=examples,
            labels=labels,
            predict_fn=llm.predict_raw_batch,
            model_id=llm.model_id,
            inference_config=llm.inference_config,
            cache=cache,
            batch_size=batch_size,
            split_name=split_name,
        )

        metrics = result["metrics"]

        individual.fitness = metrics["f1_macro"]
        individual.accuracy = metrics["accuracy"]

        individual.metadata.update({
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "n_invalid": metrics["n_invalid"],
            "invalid_rate": metrics["invalid_rate"],
            "n_examples": metrics["n_examples"],
            "n_cached": result["n_cached"],
            "n_model_texts": result["n_model_texts"],
        })

    return population


def create_next_generation(
    population: list[Individual],
    llm,
    population_size: int = 12,
    elite_count: int = 2,
    crossover_rate: float = 0.70,
    mutation_rate: float = 0.30,
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> list[Individual]:
    """Create the next generation using selection, elitism,
    crossover, and mutation.
    """

    if not population:
        raise ValueError("Population cannot be empty.")

    if population_size < 1:
        raise ValueError("Population size must be at least 1.")

    if not 0 <= elite_count <= population_size:
        raise ValueError("Elite count must be between 0 and population size.")

    if elite_count > len(population):
        raise ValueError("Elite count cannot exceed current population size.")

    if not 0 <= crossover_rate <= 1:
        raise ValueError("Crossover rate must be between 0 and 1.")

    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1.")

    if tournament_size < 1:
        raise ValueError("Tournament size must be at least 1.")

    if any(ind.fitness is None for ind in population):
        raise ValueError("Evaluate the population before reproduction.")

    if rng is None:
        rng = random.Random()

    next_generation = max(ind.generation for ind in population) + 1
    next_id = max(ind.id for ind in population) + 1

    new_population = []

    # 1. Preserve the best individuals through elitism.
    elites = select_elites(population, elite_count)

    for elite in elites:
        new_individual = Individual(
            id=next_id,
            prompt=elite.prompt,
            generation=next_generation,
            parent_ids=(elite.id,),
            fitness=elite.fitness,
            accuracy=elite.accuracy,
            metadata=copy.deepcopy(elite.metadata),
        )

        new_population.append(new_individual)
        next_id += 1

    # 2. Generate offspring until the population is full.
    while len(new_population) < population_size:

        # Select two parents using tournament selection.
        parent1 = tournament_selection(
            population,
            tournament_size,
            rng,
        )

        parent2 = tournament_selection(
            population,
            tournament_size,
            rng,
        )

        # Apply crossover according to its probability.
        if rng.random() < crossover_rate:
            child_prompt = crossover_prompts(
                llm,
                parent1.prompt,
                parent2.prompt,
            )

            # Member 2's crossover operator returns parent1 unchanged
            # if it cannot generate a valid child.
            if child_prompt == parent1.prompt:
                parent_ids = (parent1.id,)
            else:
                parent_ids = (parent1.id, parent2.id)

        else:
            child_prompt = parent1.prompt
            parent_ids = (parent1.id,)

        # Apply mutation according to its probability.
        if rng.random() < mutation_rate:
            child_prompt = mutate_prompt(
                llm,
                child_prompt,
                rng=rng,
            )

        child = Individual(
            id=next_id,
            prompt=child_prompt,
            generation=next_generation,
            parent_ids=parent_ids,
        )

        new_population.append(child)
        next_id += 1

    return new_population


def run_evolution(
    llm,
    examples,
    labels,
    population_size: int = 12,
    generations: int = 10,
    elite_count: int = 2,
    crossover_rate: float = 0.70,
    mutation_rate: float = 0.30,
    tournament_size: int = 3,
    seed_prompts: list[str] | None = None,
    seed: int = 42,
    cache: PredictionCache | None = None,
    batch_size: int | None = None,
    early_stopping_patience: int | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run the complete genetic algorithm.

    The initial population is generation 0.

    For generations=10, the algorithm evaluates generations 0 through 9.
    This means 10 evaluated populations and 9 reproduction steps.

    If output_dir is provided, optimization results are saved there.
    Otherwise, results are returned without being written to disk.
    """

    # 1. Validate configuration and input data.
    if population_size < 1:
        raise ValueError("Population size must be at least 1.")

    if generations < 1:
        raise ValueError("Generations must be at least 1.")

    if not 0 <= elite_count <= population_size:
        raise ValueError("Elite count must be between 0 and population size.")

    if tournament_size < 1:
        raise ValueError("Tournament size must be at least 1.")

    if not 0 <= crossover_rate <= 1:
        raise ValueError("Crossover rate must be between 0 and 1.")

    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1.")

    if (
        early_stopping_patience is not None
        and early_stopping_patience < 1
    ):
        raise ValueError("Early stopping patience must be at least 1.")

    if len(examples) != len(labels):
        raise ValueError("Examples and labels must have the same length.")

    if len(examples) == 0:
        raise ValueError("The evaluation dataset cannot be empty.")

    # 2. Initialize random generators and prediction cache.
    rng = random.Random(seed)

    if hasattr(llm, "set_seed"):
        llm.set_seed(seed)

    if cache is None:
        cache = PredictionCache()

    # 3. Initialize the first population.
    population = initialize_population(
        llm=llm,
        population_size=population_size,
        seed_prompts=seed_prompts,
        rng=rng,
    )

    history = []

    best_individual = None
    best_fitness = float("-inf")

    generations_without_improvement = 0

    # 4. Evolution loop.
    for generation in range(generations):

        # Evaluate every individual in the current generation.
        evaluate_population(
            population=population,
            examples=examples,
            labels=labels,
            llm=llm,
            cache=cache,
            batch_size=batch_size,
            split_name="opt",
        )

        # Calculate fitness statistics.
        fitness_scores = [
            individual.fitness
            for individual in population
        ]

        generation_best = max(
            population,
            key=lambda individual: individual.fitness,
        )

        average_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = min(fitness_scores)

        # Calculate exact textual population diversity.
        unique_prompts = len({
            individual.prompt
            for individual in population
        })

        diversity = unique_prompts / len(population)

        # Collect evaluation statistics.
        total_invalid = sum(
            individual.metadata.get("n_invalid", 0)
            for individual in population
        )

        # Number of uncached example texts evaluated,
        # not the number of physical model calls.
        total_model_texts = sum(
            individual.metadata.get("n_model_texts", 0)
            for individual in population
        )

        # Track the best individual across all generations.
        if generation_best.fitness > best_fitness:
            best_fitness = generation_best.fitness
            best_individual = copy.deepcopy(generation_best)
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Save this generation's statistics.
        generation_record = {
            "generation": generation,
            "best_fitness": generation_best.fitness,
            "average_fitness": average_fitness,
            "worst_fitness": worst_fitness,
            "best_prompt": generation_best.prompt,
            "best_accuracy": generation_best.accuracy,
            "diversity": diversity,
            "total_invalid": total_invalid,
            "total_model_texts": total_model_texts,
        }

        history.append(generation_record)

        print(
            f"Generation {generation}: "
            f"Best={generation_best.fitness:.4f}, "
            f"Average={average_fitness:.4f}, "
            f"Worst={worst_fitness:.4f}, "
            f"Diversity={diversity:.2f}"
        )

        # Stop if there has been no improvement for the
        # configured number of consecutive generations.
        if (
            early_stopping_patience is not None
            and generations_without_improvement >= early_stopping_patience
        ):
            print("Early stopping: no improvement.")
            break

        # Do not create another generation after the final evaluation.
        if generation == generations - 1:
            break

        # Create the next generation.
        population = create_next_generation(
            population=population,
            llm=llm,
            population_size=population_size,
            elite_count=elite_count,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            rng=rng,
        )

    # 5. Prepare final results.
    result = {
        "best_individual": best_individual,
        "history": history,
        "final_population": population,
    }

    # 6. Save results only if the caller provided a directory.
    if output_dir is not None:
        result["saved_files"] = save_optimization_results(
            result,
            output_dir=output_dir,
        )

    return result
