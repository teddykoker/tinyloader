# tinyloader

A tiny multiprocess data loader in ~100 lines, inspired by
[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
[geohot/tinygrad](https://github.com/geohot/tinygrad), and [karpathy/micrograd](https://github.com/karpathy/micrograd).

See blog post: [DataLoaders Explained: Building a Multi-Process Data Loader from Scratch](https://teddykoker.com/2020/12/dataloader/)

## Example

```python
from dataloader import DataLoader
import numpy as np

class Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return np.zeros((3, 32, 32)), 1


ds = Dataset(1024)
dl = DataLoader(ds, num_workers=4, batch_size=64)

x, y = next(dl)

print(x.shape)  # (64, 3, 32, 32)
print(y.shape)  # (64,)
```

## Same Example in PyTorch

```python
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return np.zeros((3, 32, 32)), 1


ds = Dataset(1024)
dl = data.DataLoader(ds, num_workers=4, batch_size=64)

x, y = next(iter(dl))

print(x.shape)  # torch.Size([64, 3, 32, 32])
print(y.shape)  # torch.Size([64])
```
