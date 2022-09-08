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
N = 3
NN = 0.9
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
        max = 0
        for i in self.shuffled_data:
            if len(i) != 39:
                break
            r = self._read(i)
            # if took > 0 and r[16] - N < average / took and np.random.random() < NN:
            #     continue
            average += r[16]
            if r[16] > max:
                max = r[16]
            took += 1
            yield np.array(r[:16], dtype=np.float32), np.array([r[16]], dtype=np.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(16, 8)
        self.fc1 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = F.relu(x)
        x = F.silu(self.fc1(x))
        return torch.clip(x, min=0)

def partition_and_shuffle(data: bytes, every: int):
    split = [data[i:i + every] for i in range(0, len(data), every)]
    shuffle(split)
    return split

def train(net):
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
            y.unsqueeze(0)
            t = torch.tensor(t)
            # print(t)

            optimizer.zero_grad()
            out = net(t)

            l = loss(out, y)

            l.backward()
            optimizer.step()

            running_loss += l.cpu().detach()
            a += 1
            if a % 2000 == 0:
                print(running_loss / 2000)
                running_loss = torch.tensor(np.array([0.0])).detach()

net = Net()
net.cuda()

train(net)

f = open("poop.h", 'w+')
l = []
# for param_tensor in net.state_dict():
#     l.append(net.state_dict()[param_tensor])
def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p).flatten() for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).unsqueeze(0).unsqueeze(0).view(-1, 1)
    return {"params": flat, "indices": indices}

a = nn.utils.parameters_to_vector(net.parameters())
open("poop.h", "w+").write("const float parameters[{}] = {{ {} }};".format(a.size()[0], str(a.tolist())[0:-1]))

