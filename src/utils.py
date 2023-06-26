import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from objects.dataset import DFGANDataset


def create_loader(imsize: int, batch_size: int, data_dir: str, split: str) -> DataLoader:
    assert split in ["train", "test"], "Wrong split type, expected train or test"
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])

    dataset = DFGANDataset(data_dir, split, image_transform)

    return DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)


def fix_seed(seed: int = 123321):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Seed {seed} fixed")


def plot_losses(g_losses: List[float], d_losses: List[float], d_gp_losses: List[float],
                path_save: str = "losses.png"):
    plt.style.use("seaborn")

    plt.figure(dpi=256)

    plt.plot(g_losses, label="G loss")
    plt.plot(d_losses, label="D loss")
    plt.plot(d_gp_losses, label="D MA-GP loss")

    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")

    plt.legend()

    plt.title("DF-GAN losses")

    plt.tight_layout()
    plt.savefig(path_save)
    plt.show()


def plot_metrics(fid: List[float], iscore: List[float], epochs: Tuple[int],
                 path_save: str = "metrics.png"):
    plt.style.use("seaborn")
    
    plt.figure(dpi=256)

    plt.plot(fid, label="FID")
    plt.plot(iscore, label="Inception Score")

    plt.xticks(np.arange(len(epochs)), epochs)

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")

    plt.legend()

    plt.title("Deep Fusion GAN metrics values per epochs")

    plt.tight_layout()
    plt.savefig(path_save)
    plt.show()
