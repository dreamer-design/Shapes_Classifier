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

from ShapeDataset import ShapeDataset
from shape_dataset_files import load_shape_dataset
from model_utils import save_model_state

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

# generate synth dataset
# generate_shape_dataset(root_dir="data/shapes", num_per_class=500)
dataset, train_loader = load_shape_dataset()

print("Classes:", dataset.classes)
print("Class indices:", dataset.class_to_idx)

# instantiate model
model = ShapeCNN(num_classes=len(dataset.classes))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# train
for epoch in range(50):
    total_loss = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

# test
# use internal class generator to load test data
# todo add out of distribution (octagons)
testSet = ShapeDataset(num_samples=2000, img_size=64)
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# eval
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

save_model_state(model, "4classes_2000samp_50epochs.pth") # torch.save(model.state_dict(), path)

