import torch
from datasets import load_dataset
from src.const import cache_dir
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, dataset, device=None, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train = None
        self.test = None
        self.val = None
        
    def loadDataset(self):
        data = load_dataset('glue', self.dataset, cache_dir=cache_dir)
        self.train = DataLoader(data['train'], batch_size=self.batch_size, shuffle=True)
        self.test = DataLoader(data['test'], batch_size=self.batch_size, shuffle=False)
        self.val = DataLoader(data['validation'], batch_size=self.batch_size, shuffle=False)

