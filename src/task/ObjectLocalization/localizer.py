import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
import timm

class ObjectDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.xmax

        bbox = [[xmin, ymin, xmax, ymax]]

        img_path = DATA_DIR + row.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            data = self.augmentations(image=img, bboxes=bbox, class_labels=[None])
            img = data['image']
            bbox = data['bboxes'][0]

        img = torch.from_numpy(img).permute(2, 0, 1)/255.0
        bbox = torch.Tensor(bbox)

        return img, bbox


class ObjectModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=4)

    def forward(self, images, gt_bboxes=None):
        bboxes = self.backbone(images)

        if gt_bboxes != None:
            loss = nn.MSELoss()(bboxes, gt_bboxes)
            return bboxes, loss

        return bboxes

def train_fn(model, dataloader, optimizer):
    total_loss = 0.0
    model.train()

    for data in tqdm(dataloader):

        images, gt_bboxes = data
        images, gt_bboxes = images.to(DEVICE), gt_bboxes.to(DEVICE)

        gt_bboxes, loss = model(images, gt_bboxes)

        optimizer.zero_grad()
        loss.backwaard()
        optimizer.step()

        total_loss += loss

    return total_loss / len(dataloader)


def valid_fn(model, dataloader, optimizer):
        total_loss = 0.0
        model.eval()

        with torch.no_grad():
            for data in tqdm(dataloader):

                images, gt_bboxes = data
                images, gt_bboxes = images.to(DEVICE), gt_bboxes.to(DEVICE)

                gt_bboxes, loss = model(images, gt_bboxes)
                total_loss += loss

            return total_loss / len(dataloader)


if __name__ == "__main__":

    CSV_FILE = 'src/data/object-localization-dataset/train.csv'
    DATA_DIR = 'src/data/object-localization-dataset'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16
    IMG_SIZE = 140

    LR = 0.001
    EPOCHS = 40
    MODEL_NAME = 'efficientnet_b0'

    NUM_COR = 4

    df = pd.read_csv(CSV_FILE)
    train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)

    train_augs = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    valid_augs = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))



    trainset = ObjectDataset(train_df, train_augs)
    validset = ObjectDataset(valid_df, valid_augs)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)

    model = ObjectModel()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_loss = np.Inf

    for i in range(EPOCHS):
        train_loss = train_fn(model, trainloader, optimizer)
        valid_loss = valid_fn(model, validloader)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            best_valid_loss = valid_loss

        print(f'Epoch : {i+1} train loss: {train_loss} valid loss: {valid_loss}')

