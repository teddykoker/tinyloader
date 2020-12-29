import numpy as np
import pytorch_lightning as pl
from dataloader import DataLoader

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate as torch_collate


class Dataset:
    def __init__(self, size=2048):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError
        # return img, label
        return np.zeros((1, 28, 28), dtype=np.float32), 1


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.mlp(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.mlp.parameters(), lr=1e-3)


def main():
    ds = Dataset()
    dl = DataLoader(ds,collate_fn=torch_collate)

    model = Model()
    pl.Trainer().fit(model, dl)


if __name__ == "__main__":
    main()
