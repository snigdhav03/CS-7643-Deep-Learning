from src.data.datasetLoader import DatasetLoader
from src.models.opt import OPT
from src.const import cache_dir
from src.prompts.qqp import QQPPrompt
import os


def in_context_learning(model_name, dataset, batch_size=32, device='cpu', examples=16):
    os.makedirs(cache_dir, exist_ok=True)
    model = OPT(model_name, device=device)
    dataset = DatasetLoader(dataset, device=device, batch_size=batch_size)
    dataset.loadDataset()
    prompt_generator = QQPPrompt(mode='icl', example=examples)
    train_data = dataset.train
    for data in train_data:
        input, label = prompt_generator(data)
        pred = model(input)
        print(pred)