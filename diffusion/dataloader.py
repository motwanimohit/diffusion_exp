import torch
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from multiprocessing import cpu_count
    
IMG_SIZE=64
BATCH_SIZE = 128


class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path, transform = None):
        self.images = [os.path.join(root_path, file) for file in os.listdir(root_path)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image[None]

def load_transformed_dataset(root_path):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t*2) - 1) #scale between [-1, 1] - scale to (0, 2) then -1 to get [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)
    data = StanfordCars(root_path=root_path, transform=data_transform)
    # test = StanfordCars(root_path=test_root_path, transform=data_transform)
    return data
    # return torch.utils.data.ConcatDataset([train, test]) #Combine all images

def load_concatenated_dataset(train_root_path, test_root_path):
    train = load_transformed_dataset(root_path=train_root_path)
    test = load_transformed_dataset(root_path=test_root_path)

    return torch.utils.data.ConcatDataset([train, test]) #Combine all images


def show_tensor_image(image):
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t*255. ),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])

    if (len(image.shape)) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transform(image))
    
train_root_path = "data/cars_train"
test_root_path = "data/cars_test"

data = load_concatenated_dataset(train_root_path, test_root_path)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)