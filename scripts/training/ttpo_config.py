from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class TTPOConfig(TrainingArguments):
    beta: float = 0.1
    alpha: float = 0.1
