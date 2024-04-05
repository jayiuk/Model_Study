import importlib
import torch

from setup import config

train_dataset = getattr(importlib.import_module('datasets.'+config.data), 'CustomDataset')(config, config.train_data_path)
test_dataset = getattr(importlib.import_module('datasets.'+config.data), 'CustomDataset')(config, config.test_data_path)

train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size)

anc, pos, neg = next(iter(train_loader))
print(anc.size())