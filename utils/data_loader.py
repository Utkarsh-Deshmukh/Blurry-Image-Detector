from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.dim = np.shape(self.data)[1] - 1
    def __len__(self):
        return np.shape(self.data)[0]
    def __getitem__(self, ind):
        x = self.data[ind][0:self.dim] / 255.0
        y = self.data[ind][-1]
        return x, y

class TestDataset(TrainDataset):
    def __init__(self, data):
        self.data = data
        self.dim = np.shape(self.data)[1] - 1
    def __getitem__(self, ind):
        x = self.data[ind][:] / 255.0
        return x