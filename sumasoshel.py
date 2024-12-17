import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Функция загрузки и обработки CSV
def load_csv_with_custom_processing(file_path):
    print(f"Загрузка файла: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.replace(',', '.').strip() for line in file.readlines()[2:]]
    processed_data = [
        [float(value) for value in line.split(';') if value] for line in lines if line
    ]
    return np.array(processed_data)

# Обработка папки с файлами
def process_folder(folder_path):
    print("Начало обработки папки...")
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
                print(f"Обработка файла для α={alpha}°")
                data_by_alpha[alpha] = load_csv_with_custom_processing(file_path)
    print("Обработка папки завершена.")
    return data_by_alpha

# Основная часть
folder_path = input("Введите путь к папке с файлами: ")
print("Загрузка данных...")
data_by_alpha = process_folder(folder_path)

# Проверка наличия эталонных углов
if 0.0 not in data_by_alpha or 45.0 not in data_by_alpha:
    raise ValueError("Не найдены файлы для углов alpha=0 и alpha=45.")
print("Эталонные углы загружены успешно.")

# Извлечение эталонных данных
print("Извлечение эталонных данных...")
y1 = data_by_alpha[0.0][:, 3]
y2 = data_by_alpha[45.0][:, 3]
x = np.linspace(0, len(y1) - 1, len(y1))

# Подготовка данных
print("Подготовка данных для оптимизации...")
alpha_values = list(data_by_alpha.keys())
y_data = np.array([data[:, 3] for data in data_by_alpha.values()])
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Model_Graphs")
os.makedirs(output_folder, exist_ok=True)

# Модель
from scipy.optimize import minimize

def model(alpha, A, B, phi1, phi2):
    alpha_rad1 = np.radians(alpha + phi1)
    alpha_rad2 = np.radians(alpha + phi2)
    term1 = A * np.sin(2 * alpha_rad1) * y2
    term2 = B * (np.cos(2 * alpha_rad2) ** 2) * y1
    return term1, term2, term1 + term2

# Функция ошибки
def error_function(params):
    A, B, phi1, phi2 = params
    total_error = 0
    for i, alpha in enumerate(alpha_values):
        _, _, y_model = model(alpha, A, B, phi1, phi2)
        total_error += np.sum((y_data[i] - y_model) ** 2)
    return total_error

# Границы для параметров (опционально, т.к. Powell сам по себе не требует ограничений)
bounds = [(-1, 1), (-1, 1), (-90, 90), (-90, 90)]

# Начальное приближение для параметров
initial_guess = [0.5, 0.5, 0.0, 0.0]

# Запуск оптимизации методом Powell
print("Запуск оптимизации методом Powell...")
result = minimize(
    error_function,
    x0=initial_guess,
    method='Powell',
    options={'maxiter': 10000, 'disp': True}
)
print("Оптимизация завершена.")

# Вывод оптимальных параметров
A_opt, B_opt, phi1_opt, phi2_opt = result.x
print(f"Оптимальные параметры: A = {A_opt}, B = {B_opt}, phi1 = {phi1_opt}, phi2 = {phi2_opt}")


# Построение и сохранение графиков
print("Построение графиков...")
for i, alpha in enumerate(alpha_values):
    term1, term2, y_model = model(alpha, A_opt, B_opt, phi1_opt, phi2_opt)
    plt.figure()
    plt.plot(x, y_data[i], label=f"Исходные данные, α={alpha}°")
    plt.plot(x, term1, label="A*sin(2(α + φ1))*y2")
    plt.plot(x, term2, label="B*cos^2(2(α + φ2))*y1")
    plt.plot(x, y_model, label="Модель")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Сравнение для α={alpha}°")
    plt.savefig(os.path.join(output_folder, f"fit_alpha_{alpha}.png"))
    plt.close()

print("Сохранение графиков зависимости от угла α...")
y_alpha_term1 = [np.mean(model(alpha, A_opt, B_opt, phi1_opt, phi2_opt)[0]) for alpha in alpha_values]
y_alpha_term2 = [np.mean(model(alpha, A_opt, B_opt, phi1_opt, phi2_opt)[1]) for alpha in alpha_values]

plt.figure()
plt.plot(alpha_values, y_alpha_term1, label="A*sin(2(α + φ1))*y1")
plt.plot(alpha_values, y_alpha_term2, label="B*cos^2(2(α + φ2))*y2")
plt.xlabel("α (градусы)")
plt.ylabel("Среднее значение")
plt.title("Компоненты модели от угла α (с учётом φ1 и φ2)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "model_components_vs_alpha_with_phi1_phi2.png"))
plt.close()

print(f"Все графики сохранены в папку: {output_folder}")
print("Выполнение программы завершено.")
