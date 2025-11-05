import os
from PIL import Image, ImageDraw
import numpy as np
import random, math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def generate_shape_dataset(
        root_dir="data",
        num_per_class=500,
        img_size=64,
        shapes=('triangle', 'square', 'pentagon', 'hexagon')
    ):
    os.makedirs(root_dir, exist_ok=True)

    for shape_type in shapes:
        shape_dir = os.path.join(root_dir, shape_type)
        os.makedirs(shape_dir, exist_ok=True)
        for i in range(num_per_class):
            img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            n_sides = {'triangle':3, 'square':4, 'pentagon':5, 'hexagon':6}[shape_type]
            r = random.randint(15, 25)
            cx, cy = random.randint(20, 44), random.randint(20, 44)
            angle_offset = random.random() * 2 * math.pi

            points = [
                (cx + r * math.cos(2 * math.pi * j / n_sides + angle_offset),
                 cy + r * math.sin(2 * math.pi * j / n_sides + angle_offset))
                for j in range(n_sides)
            ]

            color = tuple(np.random.randint(100, 255, size=3))
            draw.polygon(points, fill=color)

            img.save(os.path.join(shape_dir, f"{shape_type}_{i:05d}.png"))

    print(f"✅ Dataset created at {os.path.abspath(root_dir)}")
