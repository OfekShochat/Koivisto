import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from struct import unpack
from random import shuffle
import numpy as np

DATA_LAYOUT = "=ccicchciicciciici"
SIZE = 39
EPOCHS = 10
N = 5
NN = 0.7
#(b'\x00', b'\x00', 0, b'\x00', b'\x00', 1, b'\x01', 7, 3, b'\x01', b'\x00', -180, b'\x01', 268, 246, b'\x00', 5)
class Dataset(IterableDataset):
    def __init__(self, path: str) -> None:
        file = open(path, "rb").read()
        self.shuffled_data = partition_and_shuffle(file, SIZE)

    def _read(self, d: bytes):
        d = unpack(DATA_LAYOUT, d)
        return [ord(i) if isinstance(i, bytes) else i for i in d]

    def __iter__(self):
        average = 0
        took = 0
        for i in self.shuffled_data:
            if len(i) != 39:
                break
            r = self._read(i)
            if took > 0 and r[16] - N < average / took and np.random.random() < NN:
                continue
            average += r[16]
            took += 1
            yield np.array(r[:16], dtype=np.float32), np.array(r[16], dtype=np.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(16, 8)
        self.fc1 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = F.relu(x)
        return self.fc1(x)

def partition_and_shuffle(data: bytes, every: int):
    split = [data[i:i + every] for i in range(0, len(data), every)]
    shuffle(split)
    return split

def train():
    net = Net()
    net.cuda()

    optimizer = optim.Adam(net.parameters())
    loss = nn.MSELoss()

    dataset = Dataset("hey.bin")
    for e in range(EPOCHS):
        running_loss = torch.tensor(np.array([0.0])).detach()
        data = DataLoader(dataset, batch_size=64, pin_memory=True)
        a = 0
        for i, (t, y) in enumerate(data):
            t = t.cuda()
            y = y.cuda()
            # t = torch.tensor(t)
            # t.unsqueeze(1)

            optimizer.zero_grad()
            out = net(t)
            y.unsqueeze(1)

            l = loss(out, y)

            l.backward()
            optimizer.step()

            running_loss += l.cpu().detach()
            a += 1
            if a % 2000 == 0:
                print(running_loss / 2000)
                running_loss = torch.tensor(np.array([0.0])).detach()

train()
