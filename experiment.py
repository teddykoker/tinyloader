import numpy as np
import time
import multiprocessing
import queue
from itertools import cycle
import os
import matplotlib.pyplot as plt
import pickle

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


def train(dataloader, epochs=10, step_time=0.2):
    steps = 0
    start = time.time()
    for epoch in range(epochs):
        for batch in dataloader:
            # mimic forward, backward, and update step
            time.sleep(step_time)
            steps += 1

    end = time.time()
    return (end - start) / steps, step_time


def main():
    epochs = 1
    step_time = 0.1
    load_time = 0.0005
    size = 2048
    batch_size = 64

    num_workers = list(range(8 + 1))
    wall_time = []
    ds = Dataset(size=size, load_time=load_time)

    try:
        with open("data.pkl", "rb") as f:
            data = pickle.load(f)

    except Exception as e:
        print(e)
        for nw in num_workers:
            print(nw)
            if nw == 0:
                dl = NaiveDataLoader(ds, batch_size)
            else:
                dl = DataLoader(ds, num_workers=nw, batch_size=batch_size)
            wall, train_time = train(dl, epochs, step_time)
            wall_time.append(wall)

            del dl
            time.sleep(1.0)

        data = {
            "num_workers": num_workers,
            "wall_time": wall_time,
            "train_time": train_time,
        }
        with open("data.pkl", "wb") as f:
            pickle.dump(data, f)

    plt.plot(data["num_workers"], data["wall_time"], label="Total Step", color="black")

    plt.fill_between(
        data["num_workers"],
        data["wall_time"],
        data["train_time"],
        color="red",
        alpha=0.3,
        label="Data Loading",
    )
    plt.fill_between(
        data["num_workers"],
        data["train_time"],
        0,
        color="green",
        alpha=0.3,
        label="Training Step",
    )
    # plt.axvline(x=os.cpu_count(), label="Num. CPUs")
    plt.ylim(0.0)
    plt.xlabel("num_workers")
    plt.ylabel("Time (s) / Step")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
