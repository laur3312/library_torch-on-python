import torch

# 1. Создаем случайные данные (1599 бутылок, 12 параметров)
wine_data = torch.rand(1599, 12)

# Сохраняем сырые данные в файл
torch.save(wine_data, "wine-red.pt")
print("Файл wine-red.pt успешно создан!")

# 2. Загружаем датасет и перемешиваем строки
data = torch.load("wine-red.pt")
index = torch.randperm(data.shape[0])
data = data[index]

# 3. Вертикальный разрез: отделяем признаки (X) от ответов (y)
X, y = data[:, :-1], data[:, -1:]

# 4. Горизонтальный разрез: делим на обучение (1000 строк) и валидацию (599 строк)
X_train, X_val = X[:1000], X[1000:]
y_train, y_val = y[:1000], y[1000:]

# 5. Считаем среднее и разброс по обучающей выборке
mean_big = X_train.mean(0)
std_big = X_train.std(0)

# 6. Нормализуем данные по формуле
X_train = (X_train - mean_big) / std_big
X_val = (X_val - mean_big) / std_big

# 7. Консервируем чистые наборы данных в отдельные файлы
torch.save(X_train, "X_train.pt")
torch.save(y_train, "y_train.pt")
torch.save(X_val, "X_val.pt")
torch.save(y_val, "y_val.pt")
print("Все наборы данных (X_train, y_train, X_val, y_val) успешно подготовлены и сохранены!")