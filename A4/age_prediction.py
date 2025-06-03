import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

"""The approach is a CNN based regression model"""


class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)


train_path = 'content/faces_dataset/train'
train_ann = 'content/faces_dataset/train.csv'
train_dataset_whole = AgeDataset(train_path, train_ann, train=True)


test_path = 'content/faces_dataset/test'
test_ann = 'content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

train_size, val_size = 0.8, 0.2
batch_size = 124

train_dataset, val_dataset = random_split(train_dataset_whole, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 *7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
      
        x = x.view(-1, 256 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x.squeeze()

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint( model)
        elif score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)



model = CNN1()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Utilising",device)
model = model.to(device)
model_path = 'models/cnn1_mse_124.pt'
early_stopping = EarlyStopping(patience=5, path=model_path)
print("Training........")

for epoch_num in range(50):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", total=len(train_loader))

    for batch_num, (input, labels) in enumerate(progress_bar):
        (input, labels) = (input.to(device), labels.to(device))
        pred = model(input)
        loss = loss_fn(pred, labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss /= len(train_loader)
    model.eval()

    with torch.no_grad():
        val_loss = 0
        for batch_num, (words, tags) in enumerate(val_loader):
            (words, tags) = (words.to(device), tags.to(device))
            pred = model(words)
            val_loss += loss_fn(pred, tags)
        val_loss /= len(val_loader)

    if epoch_num % 1 == 0:
        print(f"Epoch {epoch_num}, Train Loss: {epoch_loss}, Val Loss: {val_loss}")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping....")
        break
print("Best val loss is",early_stopping.best_score)



###### SUBMISSION CSV FILE #####
print("Testing......")
model = CNN1()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Utilising",device)
model = model.to(device)
model_path = 'models/cnn1_mse_124.pt'
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

@torch.no_grad
def predict(loader, model):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions

preds = predict(test_loader, model)

submit = pd.read_csv('content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()

submit.to_csv('submission.csv',index=False)