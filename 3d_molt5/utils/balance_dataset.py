from datasets import Dataset
import torch
import numpy as np
import torch.utils

class BalanceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        super(BalanceDataset, self).__init__()

        self.datasets = dataset_list
        self.rt_lengths = [len(d) ** 0.25 for d in self.datasets]
        self.probs = [length / sum(self.rt_lengths) for length in self.rt_lengths]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_index = np.random.choice(len(self.datasets), p=self.probs)
        dataset = self.datasets[dataset_index]
        index = torch.randint(len(dataset), size=(1,)).item()

        return dataset[index]