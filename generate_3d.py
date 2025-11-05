# shapes3d_dataset.py
import os
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")  # Force Agg backend for off-screen rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset
from PIL import Image
import torch

class Shapes3DDataset(Dataset):
    def __init__(self, num_images=1000, image_size=64, save_dir=None, classes=None):
        self.num_images = num_images
        self.image_size = image_size
        self.save_dir = save_dir
        self.classes = classes or ["cube", "sphere", "cylinder", "cone"]
        if save_dir:
            for c in self.classes:
                os.makedirs(os.path.join(save_dir, c), exist_ok=True)
        self.data = []
        self._generate()

    def _generate(self):
        for i in range(self.num_images):
            cls = np.random.choice(self.classes)
            img = self._render_shape(cls)
            if self.save_dir:
                fname = os.path.join(self.save_dir, cls, f"{i:04d}.png")
                img.save(fname)
            self.data.append((img, self.classes.index(cls)))

    def _render_shape(self, shape):
        fig = plt.figure(figsize=(1,1))
        ax = fig.add_subplot(111, projection='3d')
        ax.axis("off")

        # Randomize camera
        elev = np.random.uniform(10, 80)
        azim = np.random.uniform(0, 360)
        ax.view_init(elev=elev, azim=azim)

        # Random color
        color = np.random.rand(3,)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        r = 1

        if shape == "sphere":
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x, y, z, color=color, shade=True)

        elif shape == "cube":
            r = [-1, 1]
            for s, e in zip(np.array([[x, y, z] for x in r for y in r for z in r]),
                            np.array([[x, y, z] for x in r for y in r for z in r])[1:]):
                ax.plot3D(*zip(s, e), color=color)

        elif shape == "cylinder":
            z = np.linspace(-1, 1, 30)
            theta = np.linspace(0, 2*np.pi, 30)
            theta, z = np.meshgrid(theta, z)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot_surface(x, y, z, color=color, shade=True)

        elif shape == "cone":
            height = 1.5
            radius = 1
            z = np.linspace(0, height, 30)
            theta = np.linspace(0, 2*np.pi, 30)
            Z, TH = np.meshgrid(z, theta)
            X = (height - Z) / height * radius * np.cos(TH)
            Y = (height - Z) / height * radius * np.sin(TH)
            ax.plot_surface(X, Y, Z, color=color, shade=True)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_facecolor((1,1,1))
        # Draw the canvas
        fig.canvas.draw()

        # Get renderer and its RGB buffer
        renderer = fig.canvas.get_renderer()
        buf = np.asarray(renderer.buffer_rgba())  # returns (H, W, 4)
        img = buf[:, :, :3]  # drop alpha channel, keep RGB

        plt.close(fig)
        img = Image.fromarray(img).resize((self.image_size, self.image_size))
        return img


    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
        return img, label

    def __len__(self):
        return len(self.data)
