from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class TTPOConfig(TrainingArguments):
    beta: float = 1.0
    alpha: float = 0.9
