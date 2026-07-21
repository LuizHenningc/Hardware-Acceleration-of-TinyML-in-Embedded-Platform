import scipy.io
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import ai8x


class SAT6Dataset(Dataset):
    def __init__(self, root_dir, d_type, transform=None):
        mat_file_path = os.path.join(root_dir, 'SAT-6', 'sat-6-full.mat')
        print(f"Carregando SAT-6 ({d_type}) de {mat_file_path}...")

        mat_data = scipy.io.loadmat(mat_file_path)

        if d_type == 'train':
            images = mat_data['train_x']
            labels = mat_data['train_y']
        else:
            images = mat_data['test_x']
            labels = mat_data['test_y']

        images_rgb = images[:, :, :4, :]
        self.images = np.transpose(images_rgb, (3, 2, 0, 1))
        self.labels = np.argmax(labels, axis=0)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label


def sat6_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    transform = transforms.Compose([ai8x.normalize(args=args)])
    train_dataset = SAT6Dataset(data_dir, 'train', transform=transform) if load_train else None
    test_dataset = SAT6Dataset(data_dir, 'test', transform=transform) if load_test else None
    return train_dataset, test_dataset


datasets = [
    {
        'name': 'sat6',
        'input': (4, 28, 28),
        'output': ('building', 'barren_land', 'trees', 'grassland', 'road', 'water'),
        'loader': sat6_get_datasets,
    }
]