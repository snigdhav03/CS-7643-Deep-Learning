from datetime import datetime
import os
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import IntervalStrategy
from .data.datasetLoader import DatasetLoader
from .models.opt import OPT
from .const import cache_dir, checkpoint_dir
from .prompts.qqp import QQPPrompt
from torch.utils.data import DataLoader


class TrainerLM:
    def __init__(self, model_name, dataset, adapter_name, batch_size=32, device=None, examples=16, checkpoint=None):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.dataset = dataset
        self.device = device
        self.training_args = self.getTrainingArgs(batch_size)
        self.datasetLoader = DatasetLoader(dataset, device=self.device, batch_size=batch_size)
        self.checkpoint = checkpoint
        self.model = self.get_model(model_name, adapter_name)
        self.trainer = self.getTrainer()

    def getTrainingArgs(self, batch_size):
        new_checkpoint = f'models--local--{self.model_name}--{self.adapter_name}'
        output_dir = f'./{checkpoint_dir}/{new_checkpoint}'
        os.makedirs(output_dir, exist_ok=True)
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=1000,
            save_steps=1000,
            evaluation_strategy="steps",
            label_names=["text", "labels"],
            save_strategy=IntervalStrategy.STEPS,
            save_total_limit=1,
            save_safetensors=False
        )

    def getTrainer(self):
        self.datasetLoader.loadDataset()
        trainer = CustomTrainer(
            dataloader=self.datasetLoader,
            model=self.model,
            args=self.training_args,
            # compute_metrics=self.compute_metrics,
        )
        return trainer

    def train(self):
        output_dir = self.training_args.output_dir
        path = os.path.join(os.getcwd(), output_dir)
        resume = None
        if output_dir and os.path.exists(path) and os.listdir(path):
            resume = True
        self.trainer.train(resume_from_checkpoint=resume)

    def get_model(self, model_name, adapter_name):
        os.makedirs(cache_dir, exist_ok=True)
        model = OPT(model_name, adapter_name, device=self.device, mode='classifier', checkpoint=self.checkpoint)
        return model

    # def compute_metrics(self, eval_preds):
    #     metric = evaluate.load("glue", self.dataset)
    #     logits, labels = eval_preds
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)


class CustomTrainer(Trainer):
    def __init__(self, dataloader, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = dataloader
        self.prompt_generator = QQPPrompt(mode='sft', example=16)

    def get_train_dataloader(self):
        return self.dataloader.train

    def get_eval_dataloader(self, eval_dataset=None):
        return self.dataloader.val if eval_dataset is None else DataLoader(eval_dataset)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return (None, None, None)
        input, label = self.prompt_generator(inputs)
        label_string = self.prompt_generator.label_to_answer(label)
        logit, loss = model(input, label_string)
        return (loss, None, None) if prediction_loss_only else (loss, logit, None)

    def compute_loss(self, model, inputs, return_outputs=False):
        input, label = self.prompt_generator(inputs)
        label_string = self.prompt_generator.label_to_answer(label)
        logit, loss = model(input, label_string)
        return loss
