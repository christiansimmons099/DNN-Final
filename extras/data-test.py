from pathlib import Path

from torchvision import datasets

# extras/ -> dnn-final/ -> dataset/
DATA_DIR = Path(__file__).parent.parent / "dataset"

ds = datasets.CIFAR100(root=DATA_DIR, train=True)
print(ds.data.shape)   # how many images and what size
print(ds.classes[:5])  # first 5 class names
print(ds[0])           # first image and its label number
