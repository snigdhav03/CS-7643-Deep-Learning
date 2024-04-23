from src.data.datasetLoader import DatasetLoader
from src.models.opt import OPT
from src.const import cache_dir
from src.prompts.qqp import QQPPrompt
import os
import torch


def in_context_learning(model_name, dataset, batch_size=32, device='cpu', examples=16):
    os.makedirs(cache_dir, exist_ok=True)
    model = OPT(model_name, device=device, mode='classifier')
    dataset = DatasetLoader(dataset, device=device, batch_size=batch_size)
    dataset.loadDataset()
    prompt_generator = QQPPrompt(mode='sft', example=examples)
    test_data = dataset.val
    model.eval()
    with torch.no_grad():
        for data in test_data:
            input, label = prompt_generator(data)
            label_string = prompt_generator.label_to_answer(label)
            logit, loss = model(input, label_string)
            print(logit, loss)
