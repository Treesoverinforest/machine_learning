import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    """Чтение данных из CSV файла"""
    try:
        data = pd.read_csv(filename)
        if data.shape[1] < 2:
            raise ValueError("Файл должен содержать минимум 2 столбца")
        return data
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def show_statistics(data, x_col, y_col):
    """Вывод статистической информации по выбранным столбцам"""
    print("="*50)
    print("СТАТИСТИЧЕСКАЯ ИНФОРМАЦИЯ ПО ДАННЫМ")
    print("="*50)
    
    columns = data.columns.tolist()
    x_name = columns[x_col]
    y_name = columns[y_col]
    
    print(f"Столбец X ({x_name}):")
    print(f"  Количество: {len(data[x_name])}")
    print(f"  Минимум: {data[x_name].min():.4f}")
    print(f"  Максимум: {data[x_name].max():.4f}")
    print(f"  Среднее: {data[x_name].mean():.4f}")
    print(f"  Стандартное отклонение: {data[x_name].std():.4f}")
    
    print(f"\nСтолбец Y ({y_name}):")
    print(f"  Количество: {len(data[y_name])}")
    print(f"  Минимум: {data[y_name].min():.4f}")
    print(f"  Максимум: {data[y_name].max():.4f}")
    print(f"  Среднее: {data[y_name].mean():.4f}")
    print(f"  Стандартное отклонение: {data[y_name].std():.4f}")
    print("="*50)

def linear_regression_mnk(x, y):
    """Реализация метода наименьших квадратов"""
    n = len(x)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = n * sum_x2 - sum_x * sum_x
    
    if abs(denominator) < 1e-10:
        print("Предупреждение: знаменатель близок к нулю, данные могут быть некорректными")
        a = 0
        b = np.mean(y)
    else:
        a = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y - a * sum_x) / n
    
    return a, b

def plot_original_points(x, y, x_label, y_label):
    """Отрисовка исходных точек (п.2)"""
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(x, y, color='blue', alpha=0.7, s=50, label='Исходные данные')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title('Исходные данные')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    return fig1, ax1

def plot_with_regression_line(x, y, a, b, x_label, y_label):
    """Отрисовка точек с линией регрессии (п.4)"""
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(x, y, color='blue', alpha=0.7, s=50, label='Исходные данные')
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b
    ax2.plot(x_line, y_line, color='red', linewidth=2, label=f'Линия регрессии: y = {a:.4f}x + {b:.4f}')
    
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title('Линейная регрессия (МНК)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    return fig2, ax2

def plot_with_error_squares(x, y, a, b, x_label, y_label):
    """Отрисовка с заштрихованными квадратами ошибок (п.5)"""
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(x, y, color='blue', alpha=0.7, s=50, label='Исходные данные')
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b
    ax3.plot(x_line, y_line, color='red', linewidth=2, label='Линия регрессии')
    
    # Отрисовка и заштриховка квадратов ошибок
    for i in range(len(x)):
        y_pred = a * x[i] + b
        error = y[i] - y_pred
        
        if error >= 0:
            rect = plt.Rectangle((x[i], y_pred), 0, error, 
                               linewidth=1, edgecolor='green', 
                               facecolor='green', alpha=0.3, hatch='///')
        else:
            rect = plt.Rectangle((x[i], y[i]), 0, abs(error), 
                               linewidth=1, edgecolor='green', 
                               facecolor='green', alpha=0.3, hatch='///')
        
        ax3.add_patch(rect)
        ax3.plot([x[i], x[i]], [y[i], y_pred], 'g--', alpha=0.5)
    
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_title('Линейная регрессия с визуализацией ошибок')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    return fig3, ax3

def select_columns(data):
    """Позволяет пользователю выбрать столбцы для X и Y"""
    columns = data.columns.tolist()
    print("\nДоступные столбцы:")
    for i, col in enumerate(columns):
        print(f"{i}: {col}")
    
    while True:
        try:
            x_col = int(input(f"\nВыберите номер столбца для X (0-{len(columns)-1}): "))
            if 0 <= x_col < len(columns):
                break
            else:
                print("Неверный номер столбца!")
        except ValueError:
            print("Введите число!")
    
    while True:
        try:
            y_col = int(input(f"Выберите номер столбца для Y (0-{len(columns)-1}): "))
            if 0 <= y_col < len(columns) and y_col != x_col:
                break
            elif y_col == x_col:
                print("Столбцы X и Y должны быть разными!")
            else:
                print("Неверный номер столбца!")
        except ValueError:
            print("Введите число!")
    
    return x_col, y_col

def main():
    # Автоматически используем student_scores.csv
    filename = "student_scores.csv"
    print(f"Загрузка файла: {filename}")
    
    data = read_data(filename)
    if data is None:
        return
    
    # Выбор столбцов
    x_col, y_col = select_columns(data)
    
    # Получаем имена столбцов
    columns = data.columns.tolist()
    x_name = columns[x_col]
    y_name = columns[y_col]
    
    # Извлекаем данные
    x_data = data[x_name].values
    y_data = data[y_name].values
    
    # Показ статистики
    show_statistics(data, x_col, y_col)
    
    # Вычисление параметров регрессии
    a, b = linear_regression_mnk(x_data, y_data)
    print(f"\nРЕЗУЛЬТАТЫ РЕГРЕССИИ:")
    print(f"Коэффициент наклона (a): {a:.6f}")
    print(f"Свободный член (b): {b:.6f}")
    print(f"Уравнение: y = {a:.6f} * x + {b:.6f}")
    
    # Создание трех графиков
    fig1, ax1 = plot_original_points(x_data, y_data, x_name, y_name)
    fig2, ax2 = plot_with_regression_line(x_data, y_data, a, b, x_name, y_name)
    fig3, ax3 = plot_with_error_squares(x_data, y_data, a, b, x_name, y_name)
    
    # Отображаем все три окна одновременно
    plt.show()

if __name__ == "__main__":
    main()