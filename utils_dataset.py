from generate_2d import generate_shape_dataset
from generate_3d import Shapes3DDataset

# generate_shape_dataset(root_dir="data/shapes", num_per_class=500)
# print("Classes:", dataset.classes)
# print("Class indices:", dataset.class_to_idx)

dataset = Shapes3DDataset(num_images=1000, save_dir="data/shapes3d")
print("Classes:", dataset.classes)
print("Generated images:", len(dataset))
