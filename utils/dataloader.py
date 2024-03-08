import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_real = torch.FloatTensor(x_data.real)
        self.x_imag = torch.FloatTensor(x_data.imag)
        self.y_data = torch.FloatTensor(y_data)
        self.y_data = F.one_hot(self.y_data.to(torch.int64))
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        self.x_data = torch.stack([self.x_real, self.x_imag], dim=-1)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def get_dataloader(args, samples, labels, shuffle=True):
    dataset = TensorData(samples, labels)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return data_loader


def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y
