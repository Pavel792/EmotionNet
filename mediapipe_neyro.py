import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_all_landmarks(image_tensor):
    img = image_tensor.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    results = face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        points = []
        for lm in landmarks.landmark:
            points.extend([lm.x, lm.y])
        return np.array(points, dtype=np.float32)
    else:
        return np.zeros(936, dtype=np.float32)

class HybridDataset(Dataset):
    def __init__(self, image_dataset):
        self.image_dataset = image_dataset
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        landmarks = extract_all_landmarks(image)
        return image, torch.tensor(landmarks, dtype=torch.float32), label

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    landmarks = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return images, landmarks, labels

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

datasetTrain = datasets.ImageFolder(root="emotions/train", transform=transform)
datasetTest = datasets.ImageFolder(root="emotions/test", transform=transform)

trainSize = int(0.8 * len(datasetTrain))
valSize = len(datasetTrain) - trainSize

trainData, valData = random_split(datasetTrain, [trainSize, valSize])

batchSize = 32

print("Извлечение точек")
train_hybrid = HybridDataset(trainData)
val_hybrid = HybridDataset(valData)
test_hybrid = HybridDataset(datasetTest)

getTrain = DataLoader(train_hybrid, batchSize, shuffle=True, collate_fn=collate_fn)
getVal = DataLoader(val_hybrid, batchSize, collate_fn=collate_fn)
getTest = DataLoader(test_hybrid, batchSize, collate_fn=collate_fn)

class EmotionHybridCNN(nn.Module):
    def __init__(self, numClasses=7):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.25)
        
        self.cnn_layer = nn.Linear(256 * 3 * 3, 256)
        
        self.landmarks_layer1 = nn.Linear(936, 256)
        self.landmarks_layer2 = nn.Linear(256, 128)
        
        self.combined_layer1 = nn.Linear(256 + 128, 256)
        self.combined_layer2 = nn.Linear(256, numClasses)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_image, x_landmarks):
        x = self.conv1(x_image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout4(x)
        
        x = x.view(x.size(0), -1)
        x_cnn = self.relu(self.cnn_layer(x))
        
        x_lm = self.relu(self.landmarks_layer1(x_landmarks))
        x_lm = self.relu(self.landmarks_layer2(x_lm))

        combined = torch.cat([x_cnn, x_lm], dim=1)
        x = self.relu(self.combined_layer1(combined))
        x = self.dropout(x)
        x = self.combined_layer2(x)
        
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionHybridCNN(len(datasetTrain.classes)).to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

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
    for images, landmarks, labels in trainBar:
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)

        predicts = model(images, landmarks)
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
        for images, landmarks, labels in valBar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            predicts = model(images, landmarks)
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


model.eval()
testCorrect = 0
testTotal = 0

testBar = tqdm(getTest, desc='Test')
with torch.no_grad():
    for images, landmarks, labels in testBar:
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        predicts = model(images, landmarks)
        _, predicted = torch.max(predicts, 1)
        testCorrect += (predicted == labels).sum().item()
        testTotal += labels.size(0)
        
        testBar.set_postfix({'Acc': f'{testCorrect/testTotal:.4f}'})

testAccuracy = testCorrect / testTotal
print(f'\nФинальный тест: {testAccuracy:.4f} ({testCorrect}/{testTotal})')

# 0.55
