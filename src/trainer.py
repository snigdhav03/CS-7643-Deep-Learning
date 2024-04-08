import evaluate
import numpy as np
import os
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
from .data.datasetLoader import DatasetLoader


class CustomTrainer:
    def __init__(self, checkpoint='bert-base-uncased', dataset='qqp', training_args=None, device=None):
        self.datasetLoader = DatasetLoader(dataset)
        self.dataset = dataset
        # TODO: Check why caching isn't working for the AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='cache_tokenizer')
        self.training_args = training_args if training_args else TrainingArguments("test-trainer")
        self.trainer = self.getTrainer(checkpoint)
    
    def tokenize_function(self, examples):
        if self.dataset == 'qqp':
            return self.tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
        if self.dataset == 'mnli':
            return self.tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)
        return self.tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    def getTrainer(self, checkpoint):
        self.datasetLoader.loadDataset()
        
        self.training_args.set_save(strategy="steps", steps=20, total_limit=1)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

        raw_train = self.datasetLoader.train
        raw_val = self.datasetLoader.val
        tokenized_train_datasets = raw_train.map(self.tokenize_function, batched=True)
        tokenized_val_datasets = raw_val.map(self.tokenize_function, batched=True)

        trainer = Trainer(
            model,
            self.training_args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_val_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        return trainer
    
    def train(self):
        output_dir = self.training_args.output_dir  
        path = os.path.join(os.getcwd(), output_dir)
        resume = None
        if output_dir and os.path.exists(path) and os.listdir(path):
            resume = True
        self.trainer.train(resume_from_checkpoint = resume)

    def compute_metrics(self, eval_preds):
        metric = evaluate.load("glue", self.dataset)
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)