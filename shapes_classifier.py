import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import random
import math
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from dataset_2d import ShapeDataset
from utils_model import save_model_state

Eps = 50
device = "cpu"

import torch
from torch.utils.data import DataLoader, random_split

def load_shape_dataset(data_dir="data/combined", batch_size=32):
    # Load your full dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder("data/combined", transform=transform)

    # 80/20 train-test split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # For reproducibility (always same split each run)
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, test_loader


# define model
class ShapeCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def interactive_eval(model, test_loader, dataset, device='cpu'):
    model.eval()
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(imgs).argmax(dim=1)

    total = len(imgs)
    index = [0]  # mutable container for closure

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(bottom=0.25)

    def show_image(i):
        ax.clear()
        img = np.transpose(imgs[i].cpu().numpy(), (1, 2, 0))
        ax.imshow(img)
        p, t = dataset.classes[preds[i]], dataset.classes[labels[i]]
        ax.set_title(f"Index: {i}\nPredicted: {p}\nTrue: {t}")
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            index[0] = (index[0] + 1) % total
        elif event.key == 'left':
            index[0] = (index[0] - 1) % total
        show_image(index[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_image(index[0])
    plt.show()

# ______________start main__________

# generate synth dataset
# generate_shape_dataset(root_dir="data/shapes", num_per_class=500)
dataset, train_loader, test_loader = load_shape_dataset()

print("Classes:", dataset.classes)
print("Class indices:", dataset.class_to_idx)

# instantiate model
model = ShapeCNN(num_classes=len(dataset.classes))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# train
for epoch in range( Eps ):
    total_loss = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

# eval
interactive_eval(model, test_loader, dataset, device)


model.eval()
imgs, labels = next(iter(test_loader))
with torch.no_grad():
    preds = model(imgs).argmax(dim=1)

fig, axes = plt.subplots(2, 4, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    img = np.transpose(imgs[i].numpy(), (1, 2, 0))
    ax.imshow(img)
    ax.set_title(f"P:{dataset.classes[preds[i]]}\nT:{dataset.classes[labels[i]]}")
    ax.axis('off')
plt.show()

save_model_state(model, f"8classes_{Eps}epochs.pth")
# torch.save(model.state_dict(), path)

