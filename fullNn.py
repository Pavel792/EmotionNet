import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
from random import randint

def name_to_number(i: str):
    mapping = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    return mapping[i]

X_train = []
Y_train = []
X_test = []
Y_test = []

with zipfile.ZipFile('archive (1).zip', 'r') as zfl:
#   print(zfl.namelist())

  for i in zfl.namelist():
    path = i.split('/')
    emotion_num = name_to_number(path[1])

    img_data = zfl.read(i)
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img = img.resize((48, 48))
    img_array = np.array(img)
    flat_vector = img_array.flatten() / 255.0

    if path[0] == 'test':
      X_test.append(flat_vector)
      Y_test.append(emotion_num)
    else:
      X_train.append(flat_vector)
      Y_train.append(emotion_num)


Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)


class EmotionRecogniser(nn.Module):
    def __init__(self, input_size=2304, hidden1=256, hidden2=128, num_classes=7):
        super().__init__()
        
        self.first = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        self.second = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        
        self.final = nn.Linear(hidden2, num_classes)
        self.fun = nn.ReLU()
    
    def forward(self, x):
        x = self.first(x)
        x = self.bn1(x)
        x = self.fun(x)
        
        x = self.second(x)
        x = self.bn2(x)
        x = self.fun(x)
        
        x = self.final(x)
        return x


model = EmotionRecogniser()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

Y_train = Y_train.to(device)
Y_test = Y_test.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)


loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.01)

# for _ in range(10000):
#     ind = randint(0, len(X_train) - 1)
    
#     x_batch = X_train[ind].unsqueeze(0)
#     y_true = Y_train[ind].unsqueeze(0)
    
#     y_pred = model(x_batch)
#     loss = loss_fun(y_pred, y_true)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

batch_size = 64

for epoch in range(20):
    perm = torch.randperm(len(X_train))
    
    for i in range(0, len(X_train), batch_size):
        indices = perm[i:i+batch_size]
        x_batch = X_train[indices]
        y_true = Y_train[indices]
        
        y_pred = model(x_batch)
        loss = loss_fun(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i in range(len(X_test)):
        x_batch = X_test[i].unsqueeze(0)
        y_true = Y_test[i]
        
        y_pred = model(x_batch)
        predicted_class = torch.argmax(y_pred, dim=1)
        
        if predicted_class == y_true:
            correct += 1
        total += 1

accuracy = correct / total
print(f'{accuracy} ({correct}/{total})')