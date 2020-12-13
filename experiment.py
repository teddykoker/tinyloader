import numpy as np
import time
import multiprocessing
import queue
from itertools import cycle

from dataloader import DataLoader, NaiveDataLoader


class Dataset:
    def __init__(self, size=2048, load_time=0.1):
        self.size, self.load_time = size, load_time

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError
        time.sleep(self.load_time)
        # return img, label
        return np.zeros((3, 32, 32)), 1


def train(dataloader, step_time=0.2):
    start = time.time()
    batches = 0
    for batch in dataloader:
        # mimic forward, backward, and update step
        time.sleep(step_time)
        batches += 1

    end = time.time()
    print(f"\nwall time: {end - start:.4f}")
    print(f"train time: {batches * step_time:.4f}")
    print(f"waiting time: {end - start - batches * step_time:.4f}")


if __name__ == "__main__":

    ds = Dataset(size=1024, load_time=0.01)
    dl = NaiveDataLoader(ds)
    train(dl, step_time=0.2)

    print("\nmultiprocess:\n")

    dl = DataLoader(ds, num_workers=4, batch_size=64, prefetch_batches=2)
    train(dl, step_time=0.2)
    train(dl, step_time=0.2)
