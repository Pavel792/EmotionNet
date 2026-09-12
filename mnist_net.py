import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]

        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255
            

        return img, t
    
    def __len__(self):
        return self.length
    

class DigitNN(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)

        return x

model = DigitNN(28 * 28, 32, 10)

to_tensor = tfs.ToImage()
d_train = DigitDataset("dataset", transform=to_tensor)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True, drop_last=True)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss()

epochs = 2
model.train()

for _e in range(epochs):
    train_tqdm = tqdm(train_data, leave=True)

    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_fun(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


d_test = DigitDataset("dataset", train=False, transform=to_tensor)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0

model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        y = torch.argmax(y_test, dim=1)
        Q += torch.sum(p == y).item()


Q /= len(d_test)

print(Q)