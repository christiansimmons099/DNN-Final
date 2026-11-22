import random
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from torch.utils.data import DataLoader,Subset
from torchvision import datasets
from torchvision.transforms import v2
from torch import nn

from sklearn.metrics import f1_score, classification_report, accuracy_score, top_k_accuracy_score

SEED = 42
NUM_CLASSES = 100
# Per-color (R, G, B) mean and std of the CIFAR-100 training images, from extras/normalize.py
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# The dataset folder next to this file, so it works from any working directory
DATA_DIR = Path(__file__).parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def superclass_labels(train=True):
    split = "train" if train else "test"
    with open(DATA_DIR / "cifar-100-python" / split, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    with open(DATA_DIR / "cifar-100-python" / "meta", "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    return data["coarse_labels"], meta["coarse_label_names"]


def get_transforms(train): # function turns pictures into numbers (and in this cases normalized numbers)
    steps = []
    if train:
        steps +=  [
            v2.RandomCrop(32, padding=4),  # pad to 40 x 40, and cut a random piece of img
            v2.RandomHorizontalFlip(),     # mirror left - right
        ]
    steps += [
        v2.ToImage(),                              # picture to grid of numbers
        v2.ToDtype(torch.float32, scale=True),     # 0-255 to 0-1
        v2.Normalize(CIFAR100_MEAN, CIFAR100_STD), # center around 0, numbers from my normalize.py script
    ]
    return v2.Compose(steps)


# day 3 debug_size input added
def get_loaders(batch_size=128, val_size=5000, num_workers=2, debug_size = None): # splits the dataset into training, validation and test sets
    # I include two copies of the 50k training images with different transformations
    train_full = datasets.CIFAR100(root=DATA_DIR, train=True, transform=get_transforms(train=True))
    val_full = datasets.CIFAR100(root=DATA_DIR, train=True, transform=get_transforms(train=False))
    test_set = datasets.CIFAR100(root=DATA_DIR, train=False, transform=get_transforms(train=False))

    # Shuffling numbers 0-49999 the same way every time due to SEED
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(train_full), generator=generator).tolist()

    # First 5k shuffled images are validation, other 45k are training
    train_idx = order[val_size:]
    if debug_size:
        train_idx = train_idx[:debug_size]
    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, order[:val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader


def get_numpy_splits(val_size=5000): # gets same split as get_loaders but as numpy arrays for sklearn
    train = datasets.CIFAR100(root=DATA_DIR, train=True)
    test = datasets.CIFAR100(root=DATA_DIR, train=False)

    X = train.data.reshape(len(train), -1).astype(np.float32) / 255 # flattening image and scaling it between 0-1
    y = np.array(train.targets)
    X_test = test.data.reshape(len(test), -1).astype(np.float32) / 255
    y_test = np.array(test.targets)

    #shuffles the same as get_loaders (with same SEED) that way validation is the same 5k images
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED)).numpy()
    val, tr = order[:val_size], order[val_size:]
    return X[tr], y[tr], X[val], y[val], X_test, y_test


def compute_metrics(y_true, y_pred, probs=None):
    metrics = {
        "accuracy":accuracy_score(y_true, y_pred),
        "f1_macro":f1_score(y_true, y_pred, average="macro", zero_division=0)
    }
    if probs is not None:
        metrics["top5_accuracy"] = top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(NUM_CLASSES))
    return metrics


def save_results(name, y_true, y_pred, probs, train_time, infer_time, n_params):
    RESULTS_DIR.mkdir(exist_ok=True)       # make the results folder if it isn't there yet
    path = RESULTS_DIR / f"{name}.npz"
    np.savez(path, y_true=y_true, y_pred=y_pred, probs=probs,
             train_time=train_time, infer_time=infer_time, n_params=n_params, device=str(get_device()))
    print(f"Saved {path}")


def sync(device): # wait for the gpu to finish before reading the clock
    if device.type =="mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def count_param(model): # adds up all the nuber of elements in one tensor
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, loader, optimizer, loss_fn, device): # conducts one epoch of training / learning for the model
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:  # one batch of 128 at a time
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen

@torch.no_grad() # so pytorch doesn't record operations for gradients because it'll waste memory in this case (not training)
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [],[]
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(logits.softmax(dim=1).cpu())
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    return y_true, probs.argmax(axis=1), probs


def fit(model, train_loader, val_loader, optimizer, epochs, device, name, scheduler=None):
    loss_fn = nn.CrossEntropyLoss()
    RESULTS_DIR.mkdir(exist_ok=True)
    ckpt = RESULTS_DIR /f"{name}.pt"
    best_acc = -1
    train_time = 0.0
    for epoch in range(1, epochs + 1):
        sync(device)
        start = time.perf_counter()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        sync(device)
        epoch_time = time.perf_counter() - start
        train_time += epoch_time
        if scheduler:
            scheduler.step()
        y_val, pred_val, _ = predict(model, val_loader, device)
        val_acc = (y_val == pred_val).mean()
        print(f"epoch {epoch:3d} | loss {loss:.3f} | train_acc {train_acc:.3f} | "
              f"val acc {val_acc:.3f} | epoch time {epoch_time:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt))
    return train_time

def timed_predict(model, loader, device): # gives us the inference time
    predict(model, loader, device)          # this result is thrown away
    sync(device)
    start = time.perf_counter()
    y_true, y_pred, probs = predict(model, loader, device)
    sync(device)
    return y_true, y_pred, probs, time.perf_counter() - start

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    images, labels = next(iter(train_loader)) # meaning grab one batch

    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000
    assert images.shape == (128, 3, 32, 32)
    assert images.dtype == torch.float32
    assert 0 <= labels.min() and labels.max() <= 99
    print("All checks passed")

    fake_answers = np.arange(NUM_CLASSES).repeat(10)   # 0,0,...,1,1,... (1,000 fake labels)
    m = compute_metrics(fake_answers, fake_answers)
    print("perfect guesses:", m)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0
    print("Metrics check passed")

    # check for making sure validation sets match in get_loaders and get_numpy_splits
    _, _, _, y_val, _, _ = get_numpy_splits()
    loader_val = np.array(val_loader.dataset.dataset.targets)[val_loader.dataset.indices]
    assert (y_val == loader_val).all()
    print("Numpy split check passed")