import torch
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, dataset, device=None, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train = []
        self.test = []
        self.val = []
        
    def loadDataset(self):
        data = load_dataset('glue', self.dataset)
        self.train = data['train']
        self.test = data.get('test', 'test_matched')
        self.val = data.get('validation', 'validation_matched')
