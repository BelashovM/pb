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
folder_path = input("Введите путь к папке с файлами: ")
data_by_alpha = process_folder(folder_path)

# Проверка наличия эталонных углов
if 0.0 not in data_by_alpha or 45.0 not in data_by_alpha:
    raise ValueError("Не найдены файлы для углов alpha=0 и alpha=45.")

# Извлечение эталонных данных
y1 = data_by_alpha[0.0][:, 3]
y2 = data_by_alpha[45.0][:, 3]
x = np.linspace(0, len(y1) - 1, len(y1))


# Подготовка данных для остальных углов
alpha_values = list(data_by_alpha.keys())
y_data = np.array([data[:, 3] for data in data_by_alpha.values()])
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Model_Graphs")

# Модель
# Модель
# Модель
def model(alpha, A, B, offset):
    alpha_rad = np.radians(alpha + offset)  # Добавляем смещение к углу
    term1 = A * np.sin(2 * alpha_rad) * y1
    term2 = B * (np.cos(2 * alpha_rad) ** 2) * y2
    return term1, term2, term1 + term2

# Функция ошибки
def error_function(params):
    A, B, offset = params
    total_error = 0
    for i, alpha in enumerate(alpha_values):
        _, _, y_model = model(alpha, A, B, offset)
        total_error += np.sum((y_data[i] - y_model) ** 2)
    return total_error

# Оптимизация с учетом нового параметра offset
bounds = [(0, 100), (0, 100), (-90, 90)]  # Ограничения для offset: -45° до 45°
result = differential_evolution(
    error_function,
    bounds,
    maxiter=2000000,  # Увеличение числа итераций
    popsize=1000,    # Размер популяции (по умолчанию 15, можно увеличить для более точного поиска)
    tol=1e-12       # Уменьшение допустимой ошибки для выхода
)

A_opt, B_opt, offset_opt = result.x
print(f"Оптимальные параметры: A = {A_opt}, B = {B_opt}, offset = {offset_opt}")

# Построение и сохранение графиков
for i, alpha in enumerate(alpha_values):
    term1, term2, y_model = model(alpha, A_opt, B_opt, offset_opt)
    plt.figure()
    plt.plot(x, y_data[i], label=f"Исходные данные, α={alpha}°")
    plt.plot(x, term1, label="A*sin(2(α + offset))*y1")
    plt.plot(x, term2, label="B*cos^2(2(α + offset))*y2")
    plt.plot(x, y_model, label="Модель")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Сравнение для α={alpha}°")
    plt.savefig(os.path.join(output_folder, f"fit_alpha_{alpha}.png"))
    plt.close()

# Вывод зависимости от α
y_alpha_term1 = [np.mean(model(alpha, A_opt, B_opt, offset_opt)[0]) for alpha in alpha_values]
y_alpha_term2 = [np.mean(model(alpha, A_opt, B_opt, offset_opt)[1]) for alpha in alpha_values]

plt.figure()
plt.plot(alpha_values, y_alpha_term1, label="A*sin(2(α + offset))*y1")
plt.plot(alpha_values, y_alpha_term2, label="B*cos^2(2(α + offset))*y2")
plt.xlabel("α (градусы)")
plt.ylabel("Среднее значение")
plt.title("Компоненты модели от угла α (с учетом смещения)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "model_components_vs_alpha_with_offset.png"))
plt.close()

print(f"Все графики сохранены в папку: {output_folder}")

