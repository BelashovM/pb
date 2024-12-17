import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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
os.makedirs(output_folder, exist_ok=True)


# Модель с локальными поправками
def model_with_local_corrections(alpha, A, B, phi1, phi2, shum, delta_A=0, delta_B=0):
    alpha_rad1 = np.radians(alpha + phi1)
    alpha_rad2 = np.radians(alpha + phi2)
    term1 = (A + delta_A) * np.sin(2 * alpha_rad1) * y2
    term2 = (B + delta_B) * (np.cos(2 * alpha_rad2) ** 2) * y1
    return term1, term2, term1 + term2 - shum


# Функция ошибки для глобальной оптимизации
def error_function(params):
    A, B, phi1, phi2, shum = params
    total_error = 0
    for i, alpha in enumerate(alpha_values):
        _, _, y_model = model_with_local_corrections(alpha, A, B, phi1, phi2, shum)
        total_error += np.sum((y_data[i] - y_model) ** 2)
    return total_error


# Глобальная оптимизация методом Powell
print("Запуск глобальной оптимизации...")
initial_guess = [1.0, 1.0, 0.0, 0.0, 0.0]  # Начальные параметры: A, B, phi1, phi2, shum
result_global = minimize(
    error_function,
    x0=initial_guess,
    method='Powell',
    options={'maxiter': 1000, 'disp': True}
)

A_opt, B_opt, phi1_opt, phi2_opt, shum_opt = result_global.x
print(
    f"Глобальные оптимальные параметры:\nA = {A_opt}, B = {B_opt}, phi1 = {phi1_opt}, phi2 = {phi2_opt}, shum = {shum_opt}")

# Локальная оптимизация для углов 90° и 270°
critical_alphas = [90.0, 270.0]
local_results = {}

for alpha in critical_alphas:
    idx = alpha_values.index(alpha)
    y_target = y_data[idx]


    def local_error_function(local_params):
        delta_A, delta_B = local_params
        _, _, y_model = model_with_local_corrections(alpha, A_opt, B_opt, phi1_opt, phi2_opt, shum_opt, delta_A,
                                                     delta_B)
        return np.sum((y_target - y_model) ** 2)


    print(f"Запуск локальной оптимизации для α={alpha}°...")
    result_local = minimize(
        local_error_function,
        x0=[0, 0],  # Начальные значения для поправок
        method='Powell'
    )
    local_results[alpha] = result_local.x
    print(f"Локальные поправки для α={alpha}: delta_A={result_local.x[0]}, delta_B={result_local.x[1]}")

# Построение и сохранение графиков
for i, alpha in enumerate(alpha_values):
    delta_A, delta_B = local_results.get(alpha, (0, 0))  # Поправки только для 90° и 270°
    term1, term2, y_model = model_with_local_corrections(alpha, A_opt, B_opt, phi1_opt, phi2_opt, shum_opt, delta_A,
                                                         delta_B)
    plt.figure()
    plt.plot(x, y_data[i], label=f"Исходные данные, α={alpha}°")
    plt.plot(x, term1, label="A*sin(2(α + phi1))*y2")
    plt.plot(x, term2, label="B*cos^2(2(α + phi2))*y1")
    plt.plot(x, y_model, label="Модель")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Сравнение для α={alpha}°")
    plt.savefig(os.path.join(output_folder, f"fit_alpha_{alpha}.png"))
    plt.close()

# Вычисление и сохранение СКО
with open(os.path.join(output_folder, "sko_results.txt"), 'w', encoding='utf-8') as f:
    total_sko = 0
    for i, alpha in enumerate(alpha_values):
        delta_A, delta_B = local_results.get(alpha, (0, 0))
        _, _, y_model = model_with_local_corrections(alpha, A_opt, B_opt, phi1_opt, phi2_opt, shum_opt, delta_A, delta_B)
        sko = np.sqrt(np.mean((y_data[i] - y_model) ** 2))
        total_sko += sko
        f.write(f"СКО для α={alpha}°: {sko:.10f}\n")
    avg_sko = total_sko / len(alpha_values)
    f.write(f"\nСреднее СКО: {avg_sko:.10f}")


print(f"Все графики сохранены в папку: {output_folder}")
print(f"СКО сохранены в файл: {os.path.join(output_folder, 'sko_results.txt')}")
