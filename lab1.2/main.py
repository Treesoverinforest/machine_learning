import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Загрузка данных
diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target  

# Создадим DataFrame для удобства
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Первые 5 строк данных:")
print(df.head())
print("\nОписание признаков:")
print(diabetes.DESCR[:500] + "...")  # сокращённое описание

# Выберем один признак 
feature_col = 'bmi'
X_single = df[[feature_col]].values  
y = df['target'].values

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

# 1. Линейная регрессия с использованием Scikit-Learn
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test)

coef_sklearn = lr_sklearn.coef_[0]
intercept_sklearn = lr_sklearn.intercept_

print(f"\n=== Scikit-Learn ===")
print(f"Коэффициент (наклон): {coef_sklearn:.4f}")
print(f"Свободный член (пересечение): {intercept_sklearn:.4f}")


# 2. Собственная реализация 
def linear_regression_manual(X, y):
    """
    Простая линейная регрессия: y = a * x + b
    Возвращает коэффициент a и свободный член b.
    """
    X = X.flatten()  # превращаем в одномерный массив
    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_xy = np.sum(X * y)
    sum_x2 = np.sum(X ** 2)
    
    # Формулы МНК
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b

a_manual, b_manual = linear_regression_manual(X_train, y_train)


y_pred_manual = a_manual * X_test.flatten() + b_manual

print(f"\n=== Собственная реализация ===")
print(f"Коэффициент (наклон): {a_manual:.4f}")
print(f"Свободный член (пересечение): {b_manual:.4f}")


# Вывод таблицы 
results_df = pd.DataFrame({
    'Истинные значения': y_test,
    'Предсказания (Scikit-Learn)': y_pred_sklearn,
    'Предсказания (Собственная)': y_pred_manual
})

print(f"\n=== Таблица предсказаний (первые 10 строк) ===")
print(results_df.head(10))

# Построение графиков

# График 1: Scikit-Learn
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='Истинные значения', alpha=0.7)
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Регрессия (Scikit-Learn)')
plt.title('Линейная регрессия (Scikit-Learn)')
plt.xlabel(feature_col)
plt.ylabel('Прогрессия заболевания')
plt.legend()
plt.grid(True)

# График 2: Собственная реализация
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Истинные значения', alpha=0.7)
plt.plot(X_test, y_pred_manual, color='orange', linewidth=2, label='Регрессия (Собственная)')
plt.title('Линейная регрессия (Собственная реализация)')
plt.xlabel(feature_col)
plt.ylabel('Прогрессия заболевания')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()