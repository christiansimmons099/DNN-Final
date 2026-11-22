# this file is for normalizing pixels color values (bewteen 0-1) so the models learn faster 


# loading data and scaling it from 0-255 to 0-1
from pathlib import Path

import numpy as np 
from torchvision import datasets

DATA_DIR = Path(__file__).parent.parent / "dataset"

ds = datasets.CIFAR100(root=DATA_DIR, train=True) # train=True: chooses training images only
x = ds.data.astype(np.float32) / 255

print(x.shape, x.min(), x.max())
print()

# getting the average of everything just to see the number 
print(x.mean())
print()

# Averaging per color (red, blue, green)
mean = x.mean(axis=(0,1,2))
std = x.std(axis=(0,1,2))
print("mean: ", mean)
print("std: ", std)


