import matplotlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, ToTensor

matplotlib.use('agg')
import matplotlib.pyplot as plt


class AgrilPlant(Dataset):
    """
    Agrilplant dataset for PyTorch
    """

    def __init__(self, train=True):
        # Choose paths according to training parameter
        label_path = 'data/agrilplant224x224pix_{}_labels.npy'.format('train' if train else 'test')
        images_path = 'data/agrilplant224x224pix_{}_images.npy'.format('train' if train else 'test')

        self.labels = np.load(label_path) - 1  # Load the label as a numpy array (change range from 1-10 to 0-9)
        self.images = torch.from_numpy(np.load(images_path))  # Load the images as a big pytorch byte tensor
        self.images = self.images.permute(0, 3, 1, 2).float()  # Set correct order of channels and change to float tensor

        # Use this transform later for normalizing image colors
        self.data_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        img = self.data_transform(img)

        return img, label

    def show(self, idx):
        plt.imshow(self.images[idx].permute(1, 2, 0).byte())
        plt.show()
