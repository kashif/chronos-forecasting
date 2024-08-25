import ast
import itertools
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import List, Optional

import typer
from chronos import ChronosConfig, MeanScaleUniformBins
from datasets import Dataset
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from gluonts.transform import LastValueImputation
from typer_config import use_yaml_config

from ttpo import TTPODataset, has_enough_observations

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    frequencies: str,
    probabilities: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    max_missing_prop: float = 0.9,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    model_id: str = "amazon/chronos-t5-small",
    model_type: str = "seq2seq",
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_workers: int = 4,
): 

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=ast.literal_eval(tokenizer_kwargs),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    training_data_paths = ast.literal_eval(training_data_paths)
    frequencies = ast.literal_eval(frequencies)
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq=freq),
        )
        for data_path, freq in zip(training_data_paths, frequencies)
    ]

    probabilities = ast.literal_eval(probabilities) if probabilities is not None else None
    chronos_dataset = TTPODataset(
        datasets=train_datasets,
        probabilities=probabilities,
        tokenizer=chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_id=model_id,
        model_type=model_type,
        imputation_method=LastValueImputation() if model_type == "causal" else None,
        mode="training",
    )

    # create the dataset by iterating max_steps times in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        dataset = list(executor.map(lambda _: next(iter(chronos_dataset)), range(max_steps)))

    # create a HF dataset from the json
    json_dataset = Dataset.from_list(dataset)
    dataset.save_to_disk("chronos_dataset")


if __name__ == "__main__":
    app()
