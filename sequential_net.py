import torch
from torch.nn import Linear, ReLU, Sequential

X = torch.tensor([[0.4, -0.5, 0.9], [0.3, 0.2, -0.1], [0.1, 0.7, -0.2]])
model = Sequential(
    Linear(3,10),
    ReLU(),
    Linear(10,20),
    ReLU(),
    Linear(20,1),
)
print(model(X))


Y = torch.tensor([[0.4, -0.5, 0.9], [0.3, 0.2, -0.1], [0.1, 0.7, -0.2]])
model1 = Sequential(
    Linear(3, 16),
    ReLU(),
    Linear(16, 32),
    ReLU(),
    Linear(32, 5)
)
print(model1(Y))

