import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import random
import math

class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.classes = ['triangle', 'square', 'pentagon', 'hexagon']
        self.num_classes = len(self.classes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        shape_type = random.choice(self.classes)
        img = self._draw_shape(shape_type)
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        label = self.classes.index(shape_type)
        return torch.tensor(img), torch.tensor(label)

    def _draw_shape(self, shape_type):
        img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        n_sides = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6}[shape_type]

        r = random.randint(15, 25)
        cx, cy = random.randint(20, 44), random.randint(20, 44)
        angle_offset = random.random() * 2 * math.pi

        points = [
            (cx + r * math.cos(2 * math.pi * i / n_sides + angle_offset),
             cy + r * math.sin(2 * math.pi * i / n_sides + angle_offset))
            for i in range(n_sides)
        ]

        color = tuple(np.random.randint(100, 255, size=3))
        draw.polygon(points, fill=color)
        return img
