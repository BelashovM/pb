import numpy as np
import os
import matplotlib.pyplot as plt

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
def model(alpha, A, B, y1, y2):
    alpha_rad1 = np.radians(alpha)
    alpha_rad2 = np.radians(alpha)
    term1 = A * np.sin(2 * alpha_rad1) * y2
    term2 = B * (np.cos(2 * alpha_rad2) ** 2) * y1
    return term1 + term2

# Функция ошибки
def calculate_error(A, B, alpha_values, y_data, y1, y2):
    total_error = 0
    for i, alpha in enumerate(alpha_values):
        y_model = model(alpha, A, B, y1, y2)
        total_error += np.sum((y_data[i] - y_model) ** 2)
    return total_error

# Основная часть
folder_path = input("Введите путь к папке с файлами: ")
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Model_Graphs")
os.makedirs(output_folder, exist_ok=True)

data_by_alpha = process_folder(folder_path)
if 0.0 not in data_by_alpha or 45.0 not in data_by_alpha:
    raise ValueError("Не найдены файлы для углов alpha=0 и alpha=45.")

# Эталонные данные
y1 = data_by_alpha[0.0][:, 3]
y2 = data_by_alpha[45.0][:, 3]
alpha_values = list(data_by_alpha.keys())
y_data = np.array([data[:, 3] for data in data_by_alpha.values()])
x = np.linspace(0, len(y1) - 1, len(y1))

# Перебор параметров A и B с векторизацией
print("Запуск оптимизации...")
A_range = np.linspace(-1, 1, 10000)  # Шаг 0.02
B_range = np.linspace(-1, 1, 10000)

# Векторизованный расчет ошибок
errors = np.zeros((len(A_range), len(B_range)))
for i, A in enumerate(A_range):
    for j, B in enumerate(B_range):
        total_error = calculate_error(A, B, alpha_values, y_data, y1, y2)
        errors[i, j] = total_error

# Поиск минимальной ошибки
min_error_index = np.unravel_index(np.argmin(errors), errors.shape)
best_A = A_range[min_error_index[0]]
best_B = B_range[min_error_index[1]]
min_error = errors[min_error_index]

print(f"\nОптимальные параметры найдены:")
print(f"A = {best_A}, B = {best_B}, минимальная ошибка = {min_error:.10f}")

# Построение графиков
total_sko = 0
with open(os.path.join(output_folder, "sko_results.txt"), 'w', encoding='utf-8') as f:
    for i, alpha in enumerate(alpha_values):
        y_model = model(alpha, best_A, best_B, y1, y2)
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
y_alpha = [np.mean(model(alpha, best_A, best_B, y1, y2)) for alpha in alpha_values]
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
