import numpy as np
import pandas as pd

from src.data.datasetLoader import DatasetLoader
from src.models.opt import OPT
from src.const import cache_dir
from src.prompts.qqp import QQPPrompt
from src.utils.classification_stats import ClassificationStatistics
import os
import torch


def evaluate(model_name, dataset, adapter_name, mode, batch_size=32, device='cpu', examples=8, checkpoint=None):
    os.makedirs(cache_dir, exist_ok=True)
    if checkpoint is None:
        checkpoint = f'facebook/{model_name}'
    else:
        checkpoint = f'./{cache_dir}/{checkpoint}'
    model = OPT(model_name, adapter_name, device=device, mode='classifier', checkpoint=checkpoint)
    dataset = DatasetLoader(dataset, device=device, batch_size=batch_size)
    dataset.loadDataset()
    prompt_generator = QQPPrompt(mode=mode, example=examples)
    test_data = dataset.val
    model.eval()
    model_name_for_results = f'{model_name}_{adapter_name}_{mode}'
    results = {'idx': [], 'input': [], 'label': [], 'prediction': [], 'probabilities': []}
    # path = model.save(cache_dir)
    i = 0
    with torch.no_grad():
        for data in test_data:
            input, label = prompt_generator(data)
            label_string = prompt_generator.label_to_answer(label)
            prob, loss = model(input, label_string)
            results['idx'].extend(data['idx'].tolist())
            results['input'].extend(input)
            results['label'].extend(label.tolist())
            results['prediction'].extend(torch.argmax(prob, dim=1).tolist())
            results['probabilities'].extend(prob.tolist())
            i += 1
            if i == 3:
                break
    prediction_df = pd.DataFrame(results)
    res = ClassificationStatistics(model_name, model_name_for_results, np.array(results['probabilities']),
                                   np.array(results['prediction']), np.array(results['label']), prediction_df)
    res.save_results()
