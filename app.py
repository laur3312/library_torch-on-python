import torch
from torch.nn import  Sequential, Linear, ReLU,CrossEntropyLoss
from torch.optim import SGD

X = torch.load("data.pt")
y = torch.load("target.pt")
X_train, X_val = X[:100], X[100:]
y_train, y_val = y[:100], y[100:]

model = Sequential(Linear(4,16), ReLU(), Linear(16,3))
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr = 0.01)

for _ in range(100):
    y_hat = model(X_train)
    loss = loss_fn(y_hat, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

score = (model(X_val).argmax(1) == y_val).sum() / len(y_val) * 100
print(f"Точность модели: {score:.0f}%")
