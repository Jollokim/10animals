import os

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.transforms import transforms

import cv2 as cv


class AnimalsDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)

        self.transform = transforms.ToTensor()

        self.samples = []
        for cls in self.classes:
            for sample in os.listdir(f'{self.data_dir}/{cls}'):
                self.samples.append([f'{self.data_dir}/{cls}/{sample}', cls])

        self.samples = np.array(self.samples)

    def __getitem__(self, index):
        img_path, cls = self.samples[index]

        X = cv.imread(img_path, cv.IMREAD_COLOR)
        X = self.transform(X).float()

        y = torch.zeros((len(self.classes))).float()
        y[self.classes.index(cls)] = 1

        return X, y, cls

    def __len__(self):
        return len(self.samples)
    

if __name__ == '__main__':
    dataset = AnimalsDataset('data/raw-img')

    # print('dataset total samples:', len(dataset))

    # print(dataset.__getitem__(10000))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )

    for X, y, cls in dataloader:
        print(X.shape)
        print(y.shape)
        print(cls)
