
import csv
import json
import random
from unittest.mock import MagicMock

import pytest

from src.genetic_algorithm.population import (
    Individual,
    initialize_population,
)
from src.genetic_algorithm.selection import (
    tournament_selection,
    select_elites,
)
from src.genetic_algorithm.optimizer import (
    evaluate_population,
    create_next_generation,
    run_evolution,
)


# ==================================================
# 1. Individual tests
# ==================================================

def test_individual_creation():
    individual = Individual(
        id=1,
        prompt="Classify the sentiment of this review.",
        generation=0,
    )

    assert individual.id == 1
    assert individual.generation == 0
    assert individual.fitness is None
    assert individual.accuracy is None
    assert individual.parent_ids == ()
    assert individual.metadata == {}


def test_individual_metadata_is_independent():
    first = Individual(id=1, prompt="Prompt A", generation=0)
    second = Individual(id=2, prompt="Prompt B", generation=0)

    first.metadata["precision"] = 0.9

    assert "precision" not in second.metadata


# ==================================================
# 2. Population initialization tests
# ==================================================

def test_initialize_population():
    from src.llm import MockLLM

    llm = MockLLM(seed=42)

    population = initialize_population(
        llm=llm,
        population_size=4,
        rng=random.Random(42),
    )

    assert len(population) == 4
    assert all(isinstance(ind, Individual) for ind in population)
    assert [ind.id for ind in population] == [0, 1, 2, 3]
    assert all(ind.generation == 0 for ind in population)
    assert all(ind.fitness is None for ind in population)
    assert all(ind.prompt.strip() for ind in population)


def test_initialize_population_with_start_id():
    from src.llm import MockLLM

    population = initialize_population(
        llm=MockLLM(seed=42),
        population_size=3,
        rng=random.Random(42),
        start_id=10,
    )

    assert [ind.id for ind in population] == [10, 11, 12]


# ==================================================
# 3. Tournament selection tests
# ==================================================

@pytest.fixture
def evaluated_population():
    return [
        Individual(id=0, prompt="A", generation=0, fitness=0.60),
        Individual(id=1, prompt="B", generation=0, fitness=0.90),
        Individual(id=2, prompt="C", generation=0, fitness=0.75),
        Individual(id=3, prompt="D", generation=0, fitness=0.80),
    ]


def test_tournament_selection_returns_best_contestant(
    evaluated_population,
):
    winner = tournament_selection(
        evaluated_population,
        tournament_size=4,
        rng=random.Random(42),
    )

    assert winner.id == 1
    assert winner.fitness == 0.90


def test_tournament_selection_with_size_one(evaluated_population):
    winner = tournament_selection(
        evaluated_population,
        tournament_size=1,
        rng=random.Random(42),
    )

    assert winner in evaluated_population


def test_tournament_selection_rejects_empty_population():
    with pytest.raises(ValueError):
        tournament_selection([])


def test_tournament_selection_rejects_unevaluated_population():
    population = [
        Individual(id=0, prompt="A", generation=0),
    ]

    with pytest.raises(ValueError):
        tournament_selection(population)


def test_tournament_selection_rejects_invalid_size(
    evaluated_population,
):
    with pytest.raises(ValueError):
        tournament_selection(
            evaluated_population,
            tournament_size=0,
        )


# ==================================================
# 4. Elitism tests
# ==================================================

def test_select_elites_returns_best_individuals(
    evaluated_population,
):
    elites = select_elites(evaluated_population, elite_count=2)

    assert len(elites) == 2
    assert elites[0].id == 1
    assert elites[1].id == 3


def test_select_elites_with_zero_count(evaluated_population):
    elites = select_elites(evaluated_population, elite_count=0)

    assert elites == []


def test_select_elites_rejects_negative_count(
    evaluated_population,
):
    with pytest.raises(ValueError):
        select_elites(evaluated_population, elite_count=-1)


def test_select_elites_rejects_too_many_elites(
    evaluated_population,
):
    with pytest.raises(ValueError):
        select_elites(evaluated_population, elite_count=5)


def test_select_elites_rejects_unevaluated_population():
    population = [
        Individual(id=0, prompt="A", generation=0),
    ]

    with pytest.raises(ValueError):
        select_elites(population, elite_count=1)


# ==================================================
# 5. Population evaluation tests
# ==================================================

def test_evaluate_population_stores_metrics():
    population = [
        Individual(id=0, prompt="Prompt A", generation=0),
        Individual(id=1, prompt="Prompt B", generation=0),
    ]

    fake_llm = MagicMock()
    fake_llm.model_id = "test-model"
    fake_llm.inference_config = {}
    fake_llm.predict_raw_batch = MagicMock()

    fake_results = [
        {
            "metrics": {
                "f1_macro": 0.85,
                "accuracy": 0.88,
                "precision_macro": 0.84,
                "recall_macro": 0.86,
                "n_invalid": 2,
                "invalid_rate": 0.02,
                "n_examples": 100,
            },
            "n_cached": 10,
            "n_model_texts": 90,
        },
        {
            "metrics": {
                "f1_macro": 0.92,
                "accuracy": 0.93,
                "precision_macro": 0.91,
                "recall_macro": 0.92,
                "n_invalid": 1,
                "invalid_rate": 0.01,
                "n_examples": 100,
            },
            "n_cached": 20,
            "n_model_texts": 80,
        },
    ]

    with pytest.MonkeyPatch.context() as monkeypatch:
        mock_evaluate = MagicMock(side_effect=fake_results)

        monkeypatch.setattr(
            "src.genetic_algorithm.optimizer.evaluate_prompt",
            mock_evaluate,
        )

        result = evaluate_population(
            population=population,
            examples=["review 1", "review 2"],
            labels=[1, 0],
            llm=fake_llm,
        )

    assert result is population

    assert population[0].fitness == 0.85
    assert population[0].accuracy == 0.88
    assert population[0].metadata["precision_macro"] == 0.84
    assert population[0].metadata["n_invalid"] == 2
    assert population[0].metadata["n_cached"] == 10

    assert population[1].fitness == 0.92
    assert population[1].accuracy == 0.93
    assert population[1].metadata["n_model_texts"] == 80

    assert mock_evaluate.call_count == 2


def test_evaluate_population_uses_opt_split():
    population = [
        Individual(id=0, prompt="Prompt A", generation=0),
    ]

    fake_llm = MagicMock()
    fake_llm.model_id = "test-model"
    fake_llm.inference_config = {}

    fake_result = {
        "metrics": {
            "f1_macro": 0.8,
            "accuracy": 0.8,
            "precision_macro": 0.8,
            "recall_macro": 0.8,
            "n_invalid": 0,
            "invalid_rate": 0.0,
            "n_examples": 2,
        },
        "n_cached": 0,
        "n_model_texts": 2,
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        mock_evaluate = MagicMock(return_value=fake_result)

        monkeypatch.setattr(
            "src.genetic_algorithm.optimizer.evaluate_prompt",
            mock_evaluate,
        )

        evaluate_population(
            population=population,
            examples=["good", "bad"],
            labels=[1, 0],
            llm=fake_llm,
        )

    assert mock_evaluate.call_args.kwargs["split_name"] == "opt"


# ==================================================
# 6. Next-generation creation tests
# ==================================================

def test_create_next_generation_preserves_elites_and_population_size(
    evaluated_population,
):
    next_population = create_next_generation(
        population=evaluated_population,
        llm=None,
        population_size=4,
        elite_count=1,
        crossover_rate=0,
        mutation_rate=0,
        tournament_size=3,
        rng=random.Random(42),
    )

    assert len(next_population) == 4
    assert all(ind.generation == 1 for ind in next_population)

    ids = [ind.id for ind in next_population]

    assert len(ids) == len(set(ids))
    assert all(ind_id >= 4 for ind_id in ids)

    elite = next_population[0]

    assert elite.prompt == "B"
    assert elite.fitness == 0.90
    assert elite.accuracy is None
    assert elite.parent_ids == (1,)

    offspring = next_population[1:]

    assert all(ind.fitness is None for ind in offspring)
    assert all(len(ind.parent_ids) == 1 for ind in offspring)


def test_create_next_generation_copies_elite_independently(
    evaluated_population,
):
    evaluated_population[1].metadata = {
        "precision_macro": 0.95
    }

    next_population = create_next_generation(
        population=evaluated_population,
        llm=None,
        population_size=4,
        elite_count=1,
        crossover_rate=0,
        mutation_rate=0,
        tournament_size=3,
        rng=random.Random(42),
    )

    elite = next_population[0]
    original = evaluated_population[1]

    assert elite is not original
    assert elite.metadata == original.metadata

    elite.metadata["precision_macro"] = 0.50

    assert original.metadata["precision_macro"] == 0.95


def test_create_next_generation_applies_crossover_and_mutation(
    evaluated_population,
    monkeypatch,
):
    fake_crossover = MagicMock(
        side_effect=lambda llm, p1, p2: f"crossed({p1}, {p2})"
    )

    fake_mutation = MagicMock(
        side_effect=lambda llm, prompt, rng=None: f"mutated({prompt})"
    )

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.crossover_prompts",
        fake_crossover,
    )

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.mutate_prompt",
        fake_mutation,
    )

    next_population = create_next_generation(
        population=evaluated_population,
        llm=object(),
        population_size=4,
        elite_count=1,
        crossover_rate=1.0,
        mutation_rate=1.0,
        tournament_size=3,
        rng=random.Random(42),
    )

    assert fake_crossover.call_count == 3
    assert fake_mutation.call_count == 3

    offspring = next_population[1:]

    assert all(
        ind.prompt.startswith("mutated(crossed(")
        for ind in offspring
    )

    assert all(len(ind.parent_ids) == 2 for ind in offspring)


def test_create_next_generation_tracks_parent_when_crossover_fails(
    evaluated_population,
    monkeypatch,
):
    # Simulate Member 2's crossover fallback:
    # return the first parent's prompt unchanged.
    fake_crossover = MagicMock(
        side_effect=lambda llm, p1, p2: p1
    )

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.crossover_prompts",
        fake_crossover,
    )

    next_population = create_next_generation(
        population=evaluated_population,
        llm=object(),
        population_size=4,
        elite_count=0,
        crossover_rate=1.0,
        mutation_rate=0.0,
        tournament_size=3,
        rng=random.Random(42),
    )

    assert len(next_population) == 4

    for individual in next_population:
        assert len(individual.parent_ids) == 1

    assert fake_crossover.call_count == 4


def test_create_next_generation_rejects_unevaluated_population():
    population = [
        Individual(id=0, prompt="A", generation=0, fitness=0.8),
        Individual(id=1, prompt="B", generation=0),
    ]

    with pytest.raises(ValueError):
        create_next_generation(
            population=population,
            llm=None,
            population_size=2,
        )


# ==================================================
# 7. Full evolutionary loop tests
# ==================================================

def test_run_evolution_returns_expected_structure(monkeypatch):
    from src.llm import MockLLM

    fake_llm = MockLLM(seed=42)

    def fake_evaluate_population(
        population,
        examples,
        labels,
        llm,
        cache=None,
        batch_size=None,
        split_name="opt",
    ):
        for individual in population:
            individual.fitness = 0.8
            individual.accuracy = 0.85
            individual.metadata = {
                "n_invalid": 0,
                "n_model_texts": 2,
            }
        return population

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.evaluate_population",
        fake_evaluate_population,
    )

    result = run_evolution(
        llm=fake_llm,
        examples=["good", "bad"],
        labels=[1, 0],
        population_size=4,
        generations=3,
        elite_count=1,
        crossover_rate=0,
        mutation_rate=0,
        seed=42,
    )

    assert len(result["history"]) == 3
    assert len(result["final_population"]) == 4
    assert result["best_individual"] is not None
    assert result["best_individual"].fitness == 0.8
    assert result["history"][0]["generation"] == 0
    assert result["history"][-1]["generation"] == 2


def test_run_evolution_stops_early(monkeypatch):
    from src.llm import MockLLM

    fake_llm = MockLLM(seed=42)
    evaluation_count = 0

    def fake_evaluate_population(
        population,
        examples,
        labels,
        llm,
        cache=None,
        batch_size=None,
        split_name="opt",
    ):
        nonlocal evaluation_count
        evaluation_count += 1

        for individual in population:
            individual.fitness = 0.8
            individual.accuracy = 0.85
            individual.metadata = {
                "n_invalid": 0,
                "n_model_texts": 2,
            }

        return population

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.evaluate_population",
        fake_evaluate_population,
    )

    result = run_evolution(
        llm=fake_llm,
        examples=["good", "bad"],
        labels=[1, 0],
        population_size=4,
        generations=10,
        elite_count=1,
        crossover_rate=0,
        mutation_rate=0,
        early_stopping_patience=2,
        seed=42,
    )

    assert evaluation_count == 3
    assert len(result["history"]) == 3


def test_run_evolution_rejects_empty_dataset():
    from src.llm import MockLLM

    with pytest.raises(ValueError):
        run_evolution(
            llm=MockLLM(seed=42),
            examples=[],
            labels=[],
            population_size=4,
            generations=2,
        )


def test_run_evolution_rejects_mismatched_data():
    from src.llm import MockLLM

    with pytest.raises(ValueError):
        run_evolution(
            llm=MockLLM(seed=42),
            examples=["good", "bad"],
            labels=[1],
            population_size=4,
            generations=2,
        )


def test_run_evolution_does_not_save_without_output_dir(monkeypatch):
    from src.llm import MockLLM

    fake_llm = MockLLM(seed=42)

    def fake_evaluate_population(
        population,
        examples,
        labels,
        llm,
        cache=None,
        batch_size=None,
        split_name="opt",
    ):
        for individual in population:
            individual.fitness = 0.8
            individual.accuracy = 0.85
            individual.metadata = {
                "n_invalid": 0,
                "n_model_texts": 2,
            }
        return population

    mock_save = MagicMock()

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.evaluate_population",
        fake_evaluate_population,
    )

    monkeypatch.setattr(
        "src.genetic_algorithm.optimizer.save_optimization_results",
        mock_save,
    )

    result = run_evolution(
        llm=fake_llm,
        examples=["good", "bad"],
        labels=[1, 0],
        population_size=4,
        generations=1,
        seed=42,
    )

    mock_save.assert_not_called()
    assert "saved_files" not in result


# ==================================================
# 8. Optimization logging tests
# ==================================================

def test_save_optimization_results_creates_files(tmp_path):
    from src.genetic_algorithm.logging import save_optimization_results

    best_individual = Individual(
        id=5,
        prompt="Classify the sentiment accurately.",
        generation=2,
        parent_ids=(1, 3),
        fitness=0.95,
        accuracy=0.96,
        metadata={"n_invalid": 0},
    )

    result = {
        "best_individual": best_individual,
        "history": [
            {
                "generation": 0,
                "best_fitness": 0.80,
                "average_fitness": 0.70,
                "worst_fitness": 0.60,
                "best_prompt": "Prompt A",
            },
            {
                "generation": 1,
                "best_fitness": 0.95,
                "average_fitness": 0.85,
                "worst_fitness": 0.75,
                "best_prompt": "Prompt B",
            },
        ],
    }

    output_dir = tmp_path / "experiment_01"

    saved_files = save_optimization_results(
        result,
        output_dir=output_dir,
    )

    expected_files = [
        "history.csv",
        "history.json",
        "best_prompt.txt",
        "best_individual.json",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists()

    assert saved_files["csv"] == output_dir / "history.csv"
    assert saved_files["history_json"] == output_dir / "history.json"
    assert saved_files["best_prompt"] == output_dir / "best_prompt.txt"
    assert saved_files["best_individual"] == output_dir / "best_individual.json"


def test_save_optimization_results_saves_best_prompt(tmp_path):
    from src.genetic_algorithm.logging import save_optimization_results

    best_individual = Individual(
        id=2,
        prompt="You are a sentiment analysis expert.",
        generation=1,
        fitness=0.92,
    )

    result = {
        "best_individual": best_individual,
        "history": [{"generation": 0, "best_fitness": 0.92}],
    }

    output_dir = tmp_path / "experiment_01"

    save_optimization_results(result, output_dir)

    saved_prompt = (output_dir / "best_prompt.txt").read_text(
        encoding="utf-8"
    )

    assert saved_prompt == best_individual.prompt


def test_save_optimization_results_saves_individual_metadata(tmp_path):
    from src.genetic_algorithm.logging import save_optimization_results

    best_individual = Individual(
        id=7,
        prompt="Analyze the sentiment.",
        generation=3,
        parent_ids=(2, 4),
        fitness=0.94,
        accuracy=0.95,
        metadata={"n_invalid": 1, "precision_macro": 0.93},
    )

    result = {
        "best_individual": best_individual,
        "history": [{"generation": 3, "best_fitness": 0.94}],
    }

    output_dir = tmp_path / "experiment_01"

    save_optimization_results(result, output_dir)

    with (output_dir / "best_individual.json").open(
        encoding="utf-8"
    ) as file:
        saved_data = json.load(file)

    assert saved_data["id"] == 7
    assert saved_data["prompt"] == best_individual.prompt
    assert saved_data["generation"] == 3
    assert saved_data["parent_ids"] == [2, 4]
    assert saved_data["fitness"] == 0.94
    assert saved_data["accuracy"] == 0.95
    assert saved_data["metadata"]["n_invalid"] == 1


def test_save_optimization_results_saves_history(tmp_path):
    from src.genetic_algorithm.logging import save_optimization_results

    history = [
        {
            "generation": 0,
            "best_fitness": 0.80,
            "average_fitness": 0.70,
        },
        {
            "generation": 1,
            "best_fitness": 0.90,
            "average_fitness": 0.85,
        },
    ]

    best_individual = Individual(
        id=1,
        prompt="Best prompt",
        generation=1,
        fitness=0.90,
    )

    result = {
        "best_individual": best_individual,
        "history": history,
    }

    output_dir = tmp_path / "experiment_01"

    save_optimization_results(result, output_dir)

    with (output_dir / "history.json").open(
        encoding="utf-8"
    ) as file:
        saved_history = json.load(file)

    assert saved_history == history

    with (output_dir / "history.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        saved_rows = list(csv.DictReader(file))

    assert len(saved_rows) == 2
    assert saved_rows[0]["generation"] == "0"
    assert saved_rows[1]["best_fitness"] == "0.9"


def test_save_optimization_results_handles_additional_history_fields(
    tmp_path,
):
    from src.genetic_algorithm.logging import save_optimization_results

    result = {
        "best_individual": Individual(
            id=1,
            prompt="Best prompt",
            generation=1,
            fitness=0.95,
        ),
        "history": [
            {
                "generation": 0,
                "best_fitness": 0.80,
            },
            {
                "generation": 1,
                "best_fitness": 0.95,
                "diversity": 0.75,
            },
        ],
    }

    output_dir = tmp_path / "experiment_01"

    save_optimization_results(result, output_dir)

    with (output_dir / "history.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert "diversity" in reader.fieldnames
    assert rows[1]["diversity"] == "0.75"


def test_save_optimization_results_prevents_overwriting(tmp_path):
    from src.genetic_algorithm.logging import save_optimization_results

    output_dir = tmp_path / "experiment_01"
    output_dir.mkdir()

    existing_file = output_dir / "important.txt"
    existing_file.write_text("Existing experiment", encoding="utf-8")

    result = {
        "best_individual": Individual(
            id=1,
            prompt="Prompt",
            generation=0,
            fitness=0.9,
        ),
        "history": [{"generation": 0, "best_fitness": 0.9}],
    }

    with pytest.raises(FileExistsError):
        save_optimization_results(result, output_dir)

    assert existing_file.read_text(encoding="utf-8") == "Existing experiment"


def test_save_optimization_results_rejects_empty_history(tmp_path):
    from src.genetic_algorithm.logging import save_optimization_results

    result = {
        "best_individual": Individual(
            id=1,
            prompt="Prompt",
            generation=0,
            fitness=0.9,
        ),
        "history": [],
    }

    with pytest.raises(ValueError):
        save_optimization_results(result, tmp_path / "experiment_01")


def test_save_optimization_results_rejects_missing_best_individual(
    tmp_path,
):
    from src.genetic_algorithm.logging import save_optimization_results

    result = {
        "best_individual": None,
        "history": [{"generation": 0, "best_fitness": 0.9}],
    }

    with pytest.raises(ValueError):
        save_optimization_results(result, tmp_path / "experiment_01")
