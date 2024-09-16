import torch.nn.functional as F
from transformers import Trainer

from ttpo_config import TTPOConfig


class TTPOTrainer(Trainer):
    def __init__(self, args: TTPOConfig, **kwargs):
        super().__init__(args=args, **kwargs)
        self.beta = args.beta
        self.alpha = args.alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # supervised loss
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss * self.alpha

        # margin loss
        chosen_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        rejected_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Shift logits and labels for chosen outputs
        chosen_logits = chosen_outputs.logits[:, :-1, :].contiguous()
        chosen_log_probs = chosen_logits.log_softmax(-1)

        # Shift logits and labels for rejected outputs
        rejected_logits = rejected_outputs.logits[:, :-1, :].contiguous()
        rejected_log_probs = rejected_logits.log_softmax(-1)

        # Calculate ratio using shifted log probabilities
        ratio = (chosen_log_probs - rejected_log_probs) * inputs[
            "abs_metric_diff"
        ].reshape(-1, 1, 1)
        ttpo_loss = -F.logsigmoid(self.beta * ratio).mean()

        return (loss + ttpo_loss, outputs) if return_outputs else loss + ttpo_loss
