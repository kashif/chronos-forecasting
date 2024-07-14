from transformers import Trainer

from .ttpo_config import TTPOConfig


class TTPOTrainer(Trainer):
    def __init__(self, args: TTPOConfig, **kwargs):
        super().__init__(args=args, **kwargs)
        self.beta = args.beta
        self.alpha = args.alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        import pdb

        pdb.set_trace()
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs):
        loss = self.compute_loss(model, inputs)
        return loss

    def compute_metrics(self, eval_pred):
        return {"loss": eval_pred.loss}

    def eval_step(self, model, inputs):
        loss = self.compute_loss(model, inputs)
        return loss

    def prediction_step(self, *args, **kwargs):
        return super().prediction_step(*args, **kwargs)

    def post_process_function(self, *args, **kwargs):
        return super().post_process_function(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        return super().create_optimizer_and_scheduler(num_training_steps)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)
