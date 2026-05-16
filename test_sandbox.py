import torch

print("--- Пример 1: Повторение статистики ---")
a = torch.tensor([1, 2, 3])
b = torch.tensor([1.2, 3.6, 2.4])
error = (a - b) ** 2 
print("Средняя ошибка векторов:", error.mean())

a1 = torch.tensor([[0.4, 1.8], [0.1, 1.4]])
print("Среднее по осям (dim=0):", a1.mean(0))


print("\n--- Пример 2: Разрез маленькой таблицы вручную ---")
data = torch.tensor([
    [0.076, 34.0, 5.0],
    [0.098, 67.0, 5.0],
    [0.092, 54.0, 5.0],
    [0.075, 60.0, 6.0],
])
index = torch.randperm(4)
data = data[index]
train_data, val_data = data[:2], data[2:]
X_mini, y_mini = train_data[:, :-1], train_data[:, -1:]
print("Вопросы мини-таблицы:\n", X_mini)
print("Ответы мини-таблицы:\n", y_mini)


print("\n--- Пример 3: Типы данных и устройства (Dtype & Device) ---")
a = torch.tensor([1, 2, 3])
b = torch.tensor([1.2, 3.6, 2.4])
print("Тип а:", a.dtype)
print("Тип b:", b.dtype)
c = a + b
print("Результат сложения типов:", c, "| Новый тип:", c.dtype)

# Перевод в float32
a = a.to(torch.float32)
print("Тип 'а' после приведения:", a.dtype)

# Проверка CUDA (если есть видеокарта Nvidia)
if torch.cuda.is_available():
    a = a.to("cuda")
    print("Устройство тензора:", a.device)
else:
    print("CUDA недоступна, работаем на CPU")


print("\n--- Пример 4: Базовый Автодифф и кармашек .grad ---")
a = torch.tensor(2.5, requires_grad=True)
f = a ** 2
f.backward() # Запустили расчет
print("Значение в кармашке a.grad:", a.grad) # Ожидаем 5.0


print("\n--- Пример 5: Оптимизация двух параметров в цикле ---")
x = torch.randn(2, requires_grad=True)
alpha = 0.1

for i in range(50):
    f = x[0] ** 2 + (x[1] - 4) ** 2
    f.backward()
    
    with torch.no_grad():
        x -= alpha * x.grad
        x.grad.zero_()
        
    if i % 10 == 0:
        print(f"Шаг {i}: Ошибка f(x) = {f:.4f}")