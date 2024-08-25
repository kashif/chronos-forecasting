import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from chronos import ChronosTokenizer, ChronosPipeline
from gluonts.time_feature.seasonality import get_seasonality
from gluonts.evaluation.metrics import calculate_seasonal_error, mase
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

def create_dataset_with_forecaster(
    entries: List[Dict],
    tokenizer: ChronosTokenizer,
    prediction_length: int,
    model_type: str = "seq2seq",
    num_samples: int = 20,
) -> List[Dict]:
    """
    Create a dataset by sampling random windows and using a specified Chronos forecaster.

    Parameters
    ----------
    entries : List[Dict]
        List of data entries, each containing "start" and "target" attributes.
    tokenizer : ChronosTokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    prediction_length : int
        Length of the prediction window.
    model_type : str
        Type of model, either "seq2seq" or "causal".
    num_samples : int
        Number of samples for the Chronos forecaster.

    Returns
    -------
    List[Dict]
        List of formatted datasets.
    """
    def sample_random_window(target, context_length, prediction_length):
        max_start = len(target) - context_length - prediction_length
        if max_start <= 0:
            raise ValueError("Target series is too short for the given context and prediction lengths.")
        start_idx = np.random.randint(0, max_start)
        context_window = target[start_idx:start_idx + context_length]
        prediction_window = target[start_idx + context_length:start_idx + context_length + prediction_length]
        return context_window, prediction_window

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
    )
    for entry in entries:
        context_window, prediction_window = sample_random_window(
            entry["target"], tokenizer.context_length, prediction_length
        )
        past_target = torch.tensor(context_window).unsqueeze(0)
        input_ids, attention_mask, scale = tokenizer.context_input_transform(
            past_target
        )

        future_target = torch.tensor(prediction_window).unsqueeze(0)
        freq = entry["start"].freqstr
        seasonality = get_seasonality(freq)
        models = [SeasonalNaive(season_length=seasonality)]
        sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
        ds = pd.date_range(
            start=entry["start"].to_timestamp(),
            freq=entry["start"].freq,
            periods=len(entry["target"]),
        )

        ts_df = pd.DataFrame(
            {"unique_id": "item_id", "ds": ds, "y": entry["target"]}
        )
        median = ts_df.y.median()
        ts_df = ts_df.fillna(value=median)
        fcsts_df = sf.forecast(df=ts_df, h=prediction_length)

        seasonal_naive_labels, seaononal_naive_labels_mask = (
            tokenizer.label_input_transform(
                torch.tensor(fcsts_df["SeasonalNaive"].values).unsqueeze(0), scale
            )
        )
        seasonal_naive_labels[seaononal_naive_labels_mask == 0] = -100

        seasonal_error = calculate_seasonal_error(ts_df.y.values, freq, seasonality)

        mase_sesonal_naive = mase(
            np.nan_to_num(entry["target"], median),
            fcsts_df["SeasonalNaive"].values,
            seasonal_error,
        )

        chronos_forecast = pipeline.predict(
            context=past_target,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        mase_chronos = mase(
            np.nan_to_num(entry["target"], median),
            chronos_forecast.mean(dim=1).squeeze().numpy(),
            seasonal_error,
        )
        chronos_labels, chronos_labels_mask = (
            tokenizer.label_input_transform(
                chronos_forecast.mean(dim=1), scale
            )
        )
        chronos_labels[chronos_labels_mask == 0] = -100

        if mase_chronos < mase_sesonal_naive:
            chosen_labels = chronos_labels
            rejected_labels = seasonal_naive_labels
        else:
            chosen_labels = seasonal_naive_labels
            rejected_labels = chronos_labels

        datasets.append({
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": chosen_labels.squeeze(0),
            "abs_metric_diff": abs(mase_chronos - mase_sesonal_naive),
            "chosen_labels": chosen_labels.squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0),
        })

    return datasets
