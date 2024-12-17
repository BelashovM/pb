import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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

# Модель
def model(alpha, A, B, phi1, phi2, y1, y2):
    alpha_rad1 = np.radians(alpha + phi1)
    alpha_rad2 = np.radians(alpha + phi2)
    term1 = A * np.sin(2 * alpha_rad1) * y2
    term2 = B * (np.cos(2 * alpha_rad2) ** 2) * y1
    return term1 + term2

# Функция ошибки
def error_function(params, alpha_values, y_data, y1, y2):
    A, B, phi1, phi2 = params
    total_error = 0
    for i, alpha in enumerate(alpha_values):
        y_model = model(alpha, A, B, phi1, phi2, y1, y2)
        total_error += np.sum((y_data[i] - y_model) ** 2)
    return total_error

# Основная часть
folder_path = input("Введите путь к папке с файлами: ")
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Model_Graphs")
os.makedirs(output_folder, exist_ok=True)

data_by_alpha = process_folder(folder_path)
if 0.0 not in data_by_alpha or 45.0 not in data_by_alpha:
    raise ValueError("Не найдены файлы для углов alpha=0 и alpha=45.")

y1 = data_by_alpha[0.0][:, 3]
y2 = data_by_alpha[45.0][:, 3]
alpha_values = list(data_by_alpha.keys())
y_data = np.array([data[:, 3] for data in data_by_alpha.values()])
x = np.linspace(0, len(y1) - 1, len(y1))

# Глобальная оптимизация
print("Запуск глобальной оптимизации...")
bounds = [(0, 2), (0, 2), (-90, 90), (-90, 90)]  # A, B, phi1, phi2
result = differential_evolution(
    error_function, bounds, args=(alpha_values, y_data, y1, y2),
    maxiter=5000, popsize=50, tol=1e-8
)

A_opt, B_opt, phi1_opt, phi2_opt = result.x
print("Глобальные оптимальные параметры:")
print(f"A = {A_opt}, B = {B_opt}, phi1 = {phi1_opt}, phi2 = {phi2_opt}")

# Построение графиков и подсчет СКО
total_sko = 0
with open(os.path.join(output_folder, "sko_results.txt"), 'w', encoding='utf-8') as f:
    for i, alpha in enumerate(alpha_values):
        y_model = model(alpha, A_opt, B_opt, phi1_opt, phi2_opt, y1, y2)
        sko = np.sqrt(np.mean((y_data[i] - y_model) ** 2))
        total_sko += sko

        # Запись СКО в файл
        f.write(f"СКО для α={alpha}°: {sko:.10f}\n")

        # Построение графика
        plt.figure()
        plt.plot(x, y_data[i], label=f"Исходные данные, α={alpha}°")
        plt.plot(x, y_model, label="Модель")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Сравнение для α={alpha}°")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, f"fit_alpha_{alpha}.png"))
        plt.close()

    avg_sko = total_sko / len(alpha_values)
    f.write(f"\nСреднее СКО: {avg_sko:.10f}")

# График зависимости от угла
y_alpha = [np.mean(model(alpha, A_opt, B_opt, phi1_opt, phi2_opt, y1, y2)) for alpha in alpha_values]
plt.figure()
plt.plot(alpha_values, y_alpha, 'o-', label="Модель")
plt.xlabel("α (градусы)")
plt.ylabel("Среднее значение модели")
plt.title("Зависимость модели от угла α")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "model_vs_alpha.png"))
plt.close()

print(f"Все результаты сохранены в папку: {output_folder}")