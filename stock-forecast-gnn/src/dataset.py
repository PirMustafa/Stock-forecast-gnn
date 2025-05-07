import torch
from torch.utils.data import Dataset
import numpy as np

class FinancialDataset(Dataset):
    def __init__(self, price_data, sentiment_data, graph_data, sequence_length=30):
        self.price_data = price_data
        self.sentiment_data = sentiment_data
        self.graph_data = graph_data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.price_data) - self.sequence_length

    def __getitem__(self, idx):
        price_seq = self.price_data[idx:idx+self.sequence_length]
        sentiment_seq = self.sentiment_data[idx:idx+self.sequence_length]
        graph = self.graph_data[idx+self.sequence_length - 1]
        label = self.price_data[idx+self.sequence_length]
        return torch.tensor(price_seq, dtype=torch.float32), \
               torch.tensor(sentiment_seq, dtype=torch.float32), \
               torch.tensor(graph, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.float32)