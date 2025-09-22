import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error  # <-- Импортируем метрики

# Загрузка данных
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Выбираем признак 'bmi'
feature_col = 'bmi'
X_single = df[[feature_col]].values
y = df['target'].values

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

# 1. Scikit-Learn
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test)

# -----------------------------------------------
# 2. Собственная реализация
# -----------------------------------------------
def linear_regression_manual(X, y):
    X = X.flatten()
    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_xy = np.sum(X * y)
    sum_x2 = np.sum(X ** 2)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b

a_manual, b_manual = linear_regression_manual(X_train, y_train)
y_pred_manual = a_manual * X_test.flatten() + b_manual

# -----------------------------------------------
# ВЫЧИСЛЕНИЕ И ВЫВОД МЕТРИК
# -----------------------------------------------

print("=" * 60)
print("МЕТРИКИ КАЧЕСТВА МОДЕЛЕЙ")
print("=" * 60)

# Метрики для Scikit-Learn
r2_sk = r2_score(y_test, y_pred_sklearn)
mae_sk = mean_absolute_error(y_test, y_pred_sklearn)
mape_sk = mean_absolute_percentage_error(y_test, y_pred_sklearn) * 100  # переводим в проценты

print(f"\n=== Scikit-Learn ===")
print(f"R² (коэффициент детерминации): {r2_sk:.4f}")
print(f"MAE (средняя абсолютная ошибка): {mae_sk:.4f}")
print(f"MAPE (средняя абсолютная процентная ошибка): {mape_sk:.2f}%")

# Метрики для собственной реализации
r2_man = r2_score(y_test, y_pred_manual)
mae_man = mean_absolute_error(y_test, y_pred_manual)
mape_man = mean_absolute_percentage_error(y_test, y_pred_manual) * 1

from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# --- Метрики для модели Scikit-Learn ---
print("=== Метрики для модели Scikit-Learn ===")
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
mape_sklearn = mean_absolute_percentage_error(y_test, y_pred_sklearn) * 100  # переводим в проценты

print(f"MAE (Mean Absolute Error): {mae_sklearn:.4f}")
print(f"R2 (Coefficient of Determination): {r2_sklearn:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_sklearn:.2f}%")

# --- Метрики для собственной реализации ---
print("\n=== Метрики для собственной реализации ===")
mae_manual = mean_absolute_error(y_test, y_pred_manual)
r2_manual = r2_score(y_test, y_pred_manual)
mape_manual = mean_absolute_percentage_error(y_test, y_pred_manual) * 100  # переводим в проценты

print(f"MAE (Mean Absolute Error): {mae_manual:.4f}")
print(f"R2 (Coefficient of Determination): {r2_manual:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_manual:.2f}%")