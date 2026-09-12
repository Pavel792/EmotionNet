import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

datasetTrain = datasets.ImageFolder(root="emotions/train", transform=transforms)

datasetTest = datasets.ImageFolder(root="emotions/test", transform=transforms)


trainSize = int(0.8 * len(datasetTrain))
valSize = len(datasetTrain) - trainSize

trainData, valData = random_split(datasetTrain, [trainSize, valSize])

batchSize = 32

getTrain = DataLoader(trainData, batchSize, shuffle=True)
getVal = DataLoader(valData, batchSize)
getTest = DataLoader(datasetTest, batchSize)


class EmotionResNet(nn.Module):
    def __init__(self, numClasses=7):
        super().__init__()

        # Block 1
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, padding="same")
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 96, kernel_size=3, padding="same")
        self.bn1b = nn.BatchNorm2d(96)
        
        self.proj1 = nn.Conv2d(1, 96, kernel_size=1)
        self.bn_proj1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block 2
        self.conv2a = nn.Conv2d(96, 128, kernel_size=3, padding="same")
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.bn2b = nn.BatchNorm2d(256)
        
        self.proj2 = nn.Conv2d(96, 256, kernel_size=1)
        self.bn_proj2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block 3
        self.conv3a = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 512, kernel_size=3, padding="same")
        self.bn3b = nn.BatchNorm2d(512)
        
        self.proj3 = nn.Conv2d(256, 512, kernel_size=1)
        self.bn_proj3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = nn.Linear(512 * 5 * 5, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.final = nn.Linear(4096, numClasses)
        
        self.relu = nn.ReLU()

    def forward(self, x):

        identity = self.bn_proj1(self.proj1(x))
        
        out = self.relu(self.bn1a(self.conv1a(x)))
        out = self.bn1b(self.conv1b(out))
        out = self.relu(out + identity)
        x = self.pool1(out)

        identity = self.bn_proj2(self.proj2(x))
        
        out = self.relu(self.bn2a(self.conv2a(x)))
        out = self.bn2b(self.conv2b(out))
        out = self.relu(out + identity)
        x = self.pool2(out)

        identity = self.bn_proj3(self.proj3(x))
        
        out = self.relu(self.bn3a(self.conv3a(x)))
        out = self.bn3b(self.conv3b(out))
        out = self.relu(out + identity)
        x = self.pool3(out)


        x = x.view(x.size(0), -1)
        
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.final(x)
        
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionResNet(len(datasetTrain.classes)).to(device)


lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

epochs = 50

trainLosses = []
valLosses = []
trainAccs = []
valAccs = []

for _e in range(epochs):
    model.train()
    trainLoss = 0
    trainCorrect = 0
    trainTotal = 0

    trainBar = tqdm(getTrain, desc=f'Epoch {_e} Train')
    for images, labels in trainBar:
        images, labels = images.to(device), labels.to(device)

        predicts = model(images)
        loss = lossFunction(predicts, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        trainLoss += loss.item()
        _, predicted = torch.max(predicts, 1)
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

            predicts = model(images)
            loss = lossFunction(predicts, labels)

            valLoss += loss.item()
            _, predicted = torch.max(predicts, 1)
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

    scheduler.step(avgValLoss)




model.eval()
testCorrect = 0
testTotal = 0

testBar = tqdm(getTest, desc='Test')
with torch.no_grad():
    for images, labels in testBar:
        images, labels = images.to(device), labels.to(device)
        predicts = model(images)
        _, predicted = torch.max(predicts, 1)
        testCorrect += (predicted == labels).sum().item()
        testTotal += labels.size(0)
        
        testBar.set_postfix({'Acc': f'{testCorrect/testTotal:.4f}'})

testAccuracy = testCorrect / testTotal
print(f'\nФинальный тест: {testAccuracy:.4f} ({testCorrect}/{testTotal})')

# Max Accuracy without overfitting   ~60. This result was achieved without augumentation
# 