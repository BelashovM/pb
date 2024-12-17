import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Функция загрузки и обработки CSV
def load_csv_with_custom_processing(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.replace(',', '.').strip() for line in file.readlines()[2:]]
    processed_data = [
        [float(value) for value in line.split(';') if value] for line in lines if line
    ]
    return np.array(processed_data)

# Обработка папки с файлами
def process_folder(folder_path):
    data_by_alpha = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            if '_A' in file_name:
                alpha_str = file_name.split('_A')[1][:3]
                try:
                    alpha = float(alpha_str)
                except ValueError:
                    continue
                data_by_alpha[alpha] = load_csv_with_custom_processing(file_path)
    return data_by_alpha

# Основная часть
folder_path = input("Введите путь к папке ")
data_by_alpha = process_folder(folder_path)

# Извлечение эталонных данных
y1 = data_by_alpha[0.0][:, 1]
y2 = data_by_alpha[45.0][:, 1]
x = np.linspace(0, len(y1) - 1, len(y1))

# Подготовка данных для остальных углов
alpha_values = list(data_by_alpha.keys())
y_data = np.array([data[:, 1] for data in data_by_alpha.values()])

# Модель
def model(alpha, A, B):
    alpha_rad = np.radians(alpha)
    t1 = A * np.sin(2 * alpha_rad) * y1
    t2 = B * (np.cos(2 * alpha_rad) ** 2) * y2
    return t1, t2, t1 + t2

# Функция ошибки
def error_function(params):
    A, B = params
    total_error = 0
    for i, alpha in enumerate(alpha_values):
        _, _, y_model = model(alpha, A, B)
        total_error += np.sum((y_data[i] - y_model) ** 2)
    return total_error

# Оптимизация с помощью дифференциальной эволюции
bounds = [(0.1, 1), (0.1, 1)]
result = differential_evolution(
    error_function,
    bounds,
    maxiter=2000000,
    popsize=15,
    tol=1e-7
)

A_opt, B_opt = result.x
print(f"параметры: A = {A_opt}, B = {B_opt}")

# Создание папки для графиков
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Model_Graphs")
os.makedirs(output_folder, exist_ok=True)

# Построение и сохранение графиков
for i, alpha in enumerate(alpha_values):
    term1, term2, y_model = model(alpha, A_opt, B_opt)
    plt.figure()
    plt.plot(x, y_data[i], label=f"Исходные данные, α={alpha}°")
    plt.plot(x, term1, label="A*sin(2α)*y1")
    plt.plot(x, term2, label="B*cos^2(2α)*y2")
    plt.plot(x, y_model, label="Модель")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Сравнение для α={alpha}°")
    plt.savefig(os.path.join(output_folder, f"fit_alpha_{alpha}.png"))
    plt.close()

# Вывод зависимости от α
y_alpha_term1 = [np.mean(model(alpha, A_opt, B_opt)[0]) for alpha in alpha_values]
y_alpha_term2 = [np.mean(model(alpha, A_opt, B_opt)[1]) for alpha in alpha_values]

plt.figure()
plt.plot(alpha_values, y_alpha_term1, label="A*sin(2α)*y1")
plt.plot(alpha_values, y_alpha_term2, label="B*cos^2(2α)*y2")
plt.xlabel("α (градусы)")
plt.ylabel("Среднее значение")
plt.title("Компоненты модели от угла α")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "model_components_vs_alpha.png"))
plt.close()

print(f"Все графики сохранены в папку: {output_folder}")
