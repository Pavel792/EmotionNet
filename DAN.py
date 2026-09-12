import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import random

import os

class refDbDataset(Dataset):
    def __init__(self, rootDir, labelFile, train=True, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.samples = []

        with open(labelFile, 'r') as file:
            for line in file:
                line = line.strip().split()
                imageName = line[0]
                label = int(line[1]) - 1

                isTrain = imageName.startswith('train')

                if isTrain and train:
                    self.samples.append((imageName, label))
                elif not isTrain and not train:
                    self.samples.append((imageName, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ind):
        imageName, label = self.samples[ind]

        imageName = imageName.replace('.jpg', '_aligned.jpg')

        imagePath = os.path.join(self.rootDir, imageName)

        img = Image.open(imagePath).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class EmotionDAN(nn.Module):
    def __init__(self, numClasses=7, numHeads=4):
        super().__init__()

        # Block 1
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding="same")
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.bn1b = nn.BatchNorm2d(64)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj1 = nn.Conv2d(3, 64, kernel_size=1)
        self.bn_proj1 = nn.BatchNorm2d(64)

        # Block 2
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.bn2b = nn.BatchNorm2d(128)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn_proj2 = nn.BatchNorm2d(128)

        # Block 3
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn3c = nn.BatchNorm2d(256)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj3 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn_proj3 = nn.BatchNorm2d(256)

        # Block 4
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding="same")
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding="same")
        self.bn4b = nn.BatchNorm2d(512)
        self.conv4c = nn.Conv2d(512, 512, kernel_size=3, padding="same")
        self.bn4c = nn.BatchNorm2d(512)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj4 = nn.Conv2d(256, 512, kernel_size=1)
        self.bn_proj4 = nn.BatchNorm2d(512)

        # Block 5
        self.conv5a = nn.Conv2d(512, 512, kernel_size=3, padding="same")
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding="same")
        self.bn5b = nn.BatchNorm2d(512)
        self.conv5c = nn.Conv2d(512, 512, kernel_size=3, padding="same")
        self.bn5c = nn.BatchNorm2d(512)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj5 = nn.Conv2d(512, 512, kernel_size=1)
        self.bn_proj5 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        # DAN
        self.man = MAN(channels=512, headsNum=numHeads)
        self.afn = AFN(numHeads=numHeads)

        # Classifier
        self.layer1 = nn.Linear(512 * numHeads, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.final = nn.Linear(256, numClasses)

    def forward(self, x):
        # Block 1
        id = self.bn_proj1(self.proj1(x))
        out = self.relu(self.bn1a(self.conv1a(x)))
        out = self.bn1b(self.conv1b(out))
        out = self.relu(out + id)
        x = self.max1(out)

        # Block 2
        id = self.bn_proj2(self.proj2(x))
        out = self.relu(self.bn2a(self.conv2a(x)))
        out = self.bn2b(self.conv2b(out))
        out = self.relu(out + id)
        x = self.max2(out)

        # Block 3
        id = self.bn_proj3(self.proj3(x))
        out = self.relu(self.bn3a(self.conv3a(x)))
        out = self.bn3b(self.conv3b(out))
        out = self.bn3c(self.conv3c(out))
        out = self.relu(out + id)
        x = self.max3(out)

        # Block 4
        id = self.bn_proj4(self.proj4(x))
        out = self.relu(self.bn4a(self.conv4a(x)))
        out = self.bn4b(self.conv4b(out))
        out = self.bn4c(self.conv4c(out))
        out = self.relu(out + id)
        x = self.max4(out)

        # Block 5
        id = self.bn_proj5(self.proj5(x))
        out = self.relu(self.bn5a(self.conv5a(x)))
        out = self.bn5b(self.conv5b(out))
        out = self.bn5c(self.conv5c(out))
        out = self.relu(out + id)
        x = self.max5(out)  # [batch, 512, 3, 3]

        headOutputs = self.man(x)
        fused, scaled = self.afn(headOutputs)

        # Classifier
        out = self.relu(self.layer1(fused))
        out = self.dropout1(out)
        logits = self.final(out)

        return logits, scaled
    
class SpatialAttentionUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding="same")

        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding="same")
        self.conv1x3 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding="same")
        self.conv3x1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding="same")

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1x1(x)

        branch3x3 = self.conv3x3(out)
        branch1x3 = self.conv1x3(out)
        branch3x1 = self.conv3x1(out)

        summ = branch3x3 + branch1x3 + branch3x1

        return x * self.relu(summ)

class ChannelAttentionUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.averagePool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape

        avg = self.averagePool(x).view(batch, channels)
        avg = self.mlp(avg)
        
        weights = self.sigmoid(avg).view(batch, channels, 1, 1)
        
        return x * weights

class AttentionHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = SpatialAttentionUnit(channels)
        self.channel = ChannelAttentionUnit(channels)

    def forward(self, x):
        x = self.spatial(x)
        x = self.channel(x)

        return x

class MAN(nn.Module):
    def __init__(self, channels, headsNum=4):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(channels) for _ in range(headsNum)
        ])

    def forward(self, x):
        outputs = [head(x) for head in self.heads]

        return outputs

def partitionLoss(scaled, numHeads, eps=0.1):
    
    variance = torch.var(scaled, dim=1, unbiased=False)
    
    loss = torch.log(1 + numHeads / (variance + eps))
    
    return loss.mean()

class AFN(nn.Module):
    def __init__(self, numHeads=4):
        super().__init__()
        self.numHeads = numHeads
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, headOutputs):
        batch = headOutputs[0].shape[0]
        channels = headOutputs[0].shape[1]

        vectors = [self.pool(h).view(batch, channels) for h in headOutputs]

        stacked = torch.stack(vectors, dim=1)

        scaled = torch.log_softmax(stacked, dim=1)

        fused = torch.cat(vectors, dim=1)

        return fused, scaled

datasetTrain = refDbDataset(
    rootDir="ref-db-aligned",
    labelFile="ref-db-aligned/list_partition_label.txt",
    train=True,
    transform=transform
)


datasetTest = refDbDataset(
    rootDir="ref-db-aligned",
    labelFile="ref-db-aligned/list_partition_label.txt",
    train=False,
    transform=transform
)

trainSize = int(0.8 * len(datasetTrain))
valSize = len(datasetTrain) - trainSize
trainData, valData = random_split(datasetTrain, [trainSize, valSize])

batchSize = 32
getTrain = DataLoader(trainData, batchSize, shuffle=True)
getVal = DataLoader(valData, batchSize)
getTest = DataLoader(datasetTest, batchSize)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

numHeads = 4
targetLambdaPt = 0.01
 
model = EmotionDAN(numClasses=7, numHeads=numHeads).to(device)
 
 
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

warmupEpochs = 5
epochs = 50
 
trainLosses = []
valLosses = []
trainAccs = []
valAccs = []

bestValAcc = 0.0

for _e in range(epochs):
    lambdaPt = targetLambdaPt * min(1.0, _e / warmupEpochs)

    model.train()
    trainLoss = 0
    trainCorrect = 0
    trainTotal = 0
 
    trainBar = tqdm(getTrain, desc=f'Epoch {_e} Train')
    for images, labels in trainBar:
        images, labels = images.to(device), labels.to(device)
 
        logits, scaled = model(images)
 
        clsLoss = lossFunction(logits, labels)
        ptLoss = partitionLoss(scaled, numHeads)
        loss = clsLoss + lambdaPt * ptLoss
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        trainLoss += loss.item()
        _, predicted = torch.max(logits, 1)
        trainCorrect += (predicted == labels).sum().item()
        trainTotal += labels.size(0)
 
        trainBar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{trainCorrect/trainTotal:.4f}'
        })
 
    avgTrainLoss = trainLoss / len(getTrain)
    trainAccuracy = trainCorrect / trainTotal
 
    model.eval()
    valLoss = 0
    valCorrect = 0
    valTotal = 0
 
    valBar = tqdm(getVal, desc=f'Epoch {_e} Val')
    with torch.no_grad():
        for images, labels in valBar:
            images, labels = images.to(device), labels.to(device)
 
            logits, scaled = model(images)
 
            clsLoss = lossFunction(logits, labels)
            ptLoss = partitionLoss(scaled, numHeads)
            loss = clsLoss + lambdaPt * ptLoss
 
            valLoss += loss.item()
            _, predicted = torch.max(logits, 1)
            valCorrect += (predicted == labels).sum().item()
            valTotal += labels.size(0)
 
            valBar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{valCorrect/valTotal:.4f}'
            })
 
    avgValLoss = valLoss / len(getVal)
 
    valAccuracy = valCorrect / valTotal
 
    trainLosses.append(avgTrainLoss)
    valLosses.append(avgValLoss)
    trainAccs.append(trainAccuracy)
    valAccs.append(valAccuracy)
 
    print(f'Epoch {_e}: Train Loss: {avgTrainLoss:.4f}, Train Acc: {trainAccuracy:.4f} | Val Loss: {avgValLoss:.4f}, Val Acc: {valAccuracy:.4f}')
    if avgValLoss > avgTrainLoss * 1.2:
        print('Возможно переобучение!')

    if valAccuracy > bestValAcc:
        bestValAcc = valAccuracy
        torch.save(model.state_dict(), 'best_model.pt')
        print(f'Сохранена лучшая модель: Val Acc {bestValAcc:.4f}')
 
    scheduler.step(avgValLoss)
 
 
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
testCorrect = 0
testTotal = 0
 
testBar = tqdm(getTest, desc='Test')
with torch.no_grad():
    for images, labels in testBar:
        images, labels = images.to(device), labels.to(device)
        logits, scaled = model(images)
        _, predicted = torch.max(logits, 1)
        testCorrect += (predicted == labels).sum().item()
        testTotal += labels.size(0)
 
        testBar.set_postfix({'Acc': f'{testCorrect/testTotal:.4f}'})
 
testAccuracy = testCorrect / testTotal
print(f'\nФинальный тест (лучшая модель по Val Acc {bestValAcc:.4f}): {testAccuracy:.4f} ({testCorrect}/{testTotal})')

# 0.80