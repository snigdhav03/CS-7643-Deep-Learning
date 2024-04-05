import torch
import torchtext
from torchtext import data
from torchtext.datasets import MNLI, RTE, QQP

class DatasetLoader:
    def __init__(self, dataset, batch_size=64, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.loader = data.BucketIterator(dataset, batch_size=batch_size, 
                        sort_key=self.get_sort_key(dataset),
                        sort_within_batch=True,
                        device=self.device)
    
    def __iter__(self):
        return iter(self.loader)
    
    def get_sort_key(self, dataset):
        if isinstance(dataset.dataset, MNLI):
            return lambda x: len(x.premise)
        elif isinstance(dataset.dataset, RTE):
            return lambda x: len(x.premise)
        elif isinstance(dataset.dataset, QQP):
            return lambda x: len(x.question1)
        return None
