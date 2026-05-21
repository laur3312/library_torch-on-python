import torch

# Создаем случайные данные, похожие на характеристики вина
# Например: 1599 бутылок (строк) и 12 параметров (колонок)
wine_data = torch.rand(1599, 12)

# Сохраняем эти данные в файл "wine-red.pt"
torch.save(wine_data, "wine-red.pt")

print("Файл wine-red.pt успешно создан!")


a = torch.tensor([1,2,3])
b = torch.tensor([1.2, 3.6, 2.4])

error = (a-b) ** 2 
print(error.mean())


a1 = torch.tensor([[0.4, 1.8], [0.1, 1.4]])
print(a1.mean(0))



data = torch.tensor([
    [0.076, 34.0, 5.0],
    [0.098, 67.0, 5.0],
    [0.092, 54.0, 5.0],
    [0.075, 60.0, 6.0],
])
index = torch.randperm(4)
data = data[index]
train_data, val_data = data[:2], data[2:]
print(train_data)
print(val_data)
X_train, y_train = train_data[:, :-1], train_data[:, -1:]
print(X_train)
print(y_train)

mean = data.mean(0)
std = data.std(0)
print(mean)
print(std)

data = torch.load("wine-red.pt")
index = torch.randperm(data.shape[0])
data = data[index]

X, y = data[:, :-1], data[:, -1:]
X_train, X_val = X[:1000], X[1000:]
y_train, y_val = y[:1000], y[1000:]
X_train = (X_train - data.mean(0)[:-1]) / data.std(0)[:-1]
X_val = (X_val - data.mean(0)[:-1]) / data.std(0)[:-1]

torch.save(X_train, "X_train.pt")
torch.save(y_train, "y_train.pt")
torch.save(X_val, "X_val.pt")
torch.save(y_val, "y_val.pt")

a - torch.tensor([1, 2, 3])
b = torch.tensor([1.2, 3.6, 2.4])
print(a.dtype)
print(b.dtype)
c = a + b
print(c)
print(c.dtype)

a - torch.tensor([1, 2, 3])
a = a.to(torch.float32)
print(a.dtype)



a - torch.tensor([1, 2, 3])
a = a.to("cuda")
print(a.device)



a = torch.tensor(2.5, requires_grad=True)
f = a**2
print(f.backward())



# x = torch.randn(2, requires_grad=True)
# f = x[0] ** 2 + (x[1] - 4) ** 2
# print(x)
# print(f)


# x = torch.randn(2, requires_grad=True)
# alpha = 0.1

# # for _ in range(50):
# #     f = x ** 2 + (x - 4) ** 2
# #     f.backward()
    
# #     with torch.no_grad():
# #         x -= alpha * x.grad  # Чистый, короткий и правильный код!
# #         x.grad.zero_()
        
# #     print(f"f(x)={f:.4f}")


X_train = torch.load("X_train.pt")
y_train = torch.load("y_train.pt")
X_val = torch.load("X_val.pt")
y_val = torch.load("y_val.pt")
W = torch.randn(11, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

y_hat = X_train @ W + b
loss = ((y_hat - y_train)** 2 ).mean()
print(loss)
print(y_hat.shape)
print(y_train.shape)
alpha = 0.01

for _ in range(1000):
    y_hat = X_train @ W + b
    loss = ((y_hat - y_train) ** 2).mean()
    loss.backward()
    
    with torch.no_grad():
        b -= alpha * b.grad
        W -= alpha * W.grad
        
        W.grad.zero_()
        b.grad.zero_()
print(f"Ошибка {loss:.5f}")
with torch.no_grad():
    y_hat = X_val @ W + b
score = (y_hat - y_val).abs().mean()
print(f"Среднее отклонение{score:.2f} балла.")


