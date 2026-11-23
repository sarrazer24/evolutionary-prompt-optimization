
import csv
import json
import tempfile
from pathlib import Path


def save_optimization_results(
    result: dict,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save the results of a genetic algorithm run.

    Creates:
        - history.csv
        - history.json
        - best_prompt.txt
        - best_individual.json

    Raises:
        ValueError: If the results are missing or invalid.
        FileExistsError: If the destination already contains files.
    """

    if not isinstance(result, dict):
        raise ValueError("Result must be a dictionary.")

    history = result.get("history")
    best_individual = result.get("best_individual")

    if not isinstance(history, list) or not history:
        raise ValueError("Optimization history cannot be empty.")

    if not all(isinstance(record, dict) for record in history):
        raise ValueError("Every history record must be a dictionary.")

    if best_individual is None:
        raise ValueError("No best individual was provided.")

    if not isinstance(best_individual.prompt, str) or not best_individual.prompt.strip():
        raise ValueError("The best individual must have a non-empty prompt.")

    output_path = Path(output_dir)

    # Prevent accidental overwriting of an existing experiment.
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"The output directory already contains files: {output_path}"
        )

    # Collect all history fields while preserving their order.
    fieldnames = list(dict.fromkeys(
        key
        for record in history
        for key in record
    ))

    if not fieldnames:
        raise ValueError("Optimization history contains no fields.")

    # Prepare the best individual's data.
    best_data = {
        "id": best_individual.id,
        "prompt": best_individual.prompt,
        "generation": best_individual.generation,
        "parent_ids": list(best_individual.parent_ids),
        "fitness": best_individual.fitness,
        "accuracy": best_individual.accuracy,
        "metadata": best_individual.metadata,
    }

    # Prepare all output contents before writing.
    from io import StringIO

    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(history)

    history_json = json.dumps(
        history,
        indent=4,
        ensure_ascii=False,
        default=str,
    )

    individual_json = json.dumps(
        best_data,
        indent=4,
        ensure_ascii=False,
        default=str,
    )

    best_prompt = best_individual.prompt

    output_path.mkdir(parents=True, exist_ok=True)

    # Define the four output files and their contents.
    outputs = {
        "history.csv": csv_buffer.getvalue(),
        "history.json": history_json,
        "best_prompt.txt": best_prompt,
        "best_individual.json": individual_json,
    }

    final_paths = {
        "csv": output_path / "history.csv",
        "history_json": output_path / "history.json",
        "best_prompt": output_path / "best_prompt.txt",
        "best_individual": output_path / "best_individual.json",
    }

    # Write temporary files first, then replace the final files.
    temporary_paths = []

    try:
        for filename, content in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path,
                delete=False,
                suffix=".tmp",
            ) as temporary_file:
                temporary_file.write(content)
                temporary_paths.append(
                    (Path(temporary_file.name), output_path / filename)
                )

        for temporary_path, final_path in temporary_paths:
            temporary_path.replace(final_path)

    finally:
        # Clean up any temporary files left after an error.
        for temporary_path, _ in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

    return final_paths
