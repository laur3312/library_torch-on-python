import torch
from torch.nn import Linear

X = torch.tensor([[0.4, 0.5], [0.9, 0.2], [0.2, 0.0], [0.1, 0.0]])
W = torch.tensor([[0.2, 0.5, 1.0], [0.7, 0.4, 0.4]])
print(X)
print(W)
print(X @ W)
layer = Linear(2,3)
print(layer(X))