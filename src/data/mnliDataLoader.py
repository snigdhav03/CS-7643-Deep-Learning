import torch
import torchtext
from torchtext import data
from torchtext.datasets import MNLI
from datasetLoader import DatasetLoader

TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

train_qqp, val_qqp, test_qqp = MNLI.splits(TEXT, LABEL)

qqp_loader = DatasetLoader(train_qqp, batch_size=64)

for batch in qqp_loader:
    premise, hypothesis = batch.premise, batch.hypothesis
    labels = batch.label
    print(premise)
    print(labels)
