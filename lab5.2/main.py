import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Загрузка и предобработка данных 
df = pd.read_csv('diabetes.csv')

# Предварительный анализ данных
print("Первые 5 строк датасета:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())
print("\nОписательная статистика:")
print(df.describe())

# Обработка нулевых значений (0 может означать отсутствие данных для некоторых признаков)
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, df[col][df[col] != 0].median())

# Разделение на признаки и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

# Задача Случайный лес

print("\n" + "="*60)
print("ЗАДАЧА 1: СЛУЧАЙНЫЙ ЛЕС")
print("="*60)

# Исследование качества модели от глубины деревьев
print("\n1.1 Исследование зависимости качества от глубины деревьев")

max_depths = range(1, 21)
f1_scores_depth = []
train_times_depth = []

for depth in max_depths:
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_depth.append(train_time)
    
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_depth.append(f1)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(max_depths, f1_scores_depth, marker='o', linewidth=2, markersize=8)
plt.xlabel('Максимальная глубина деревьев')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины деревьев в случайном лесу')
plt.grid(True)
plt.xticks(max_depths)
plt.show()

# Определение оптимальной глубины
optimal_depth_rf = max_depths[np.argmax(f1_scores_depth)]
print(f"Оптимальная глубина деревьев: {optimal_depth_rf}")
print(f"Максимальный F1-score: {max(f1_scores_depth):.4f}")

#Исследование качества модели от количества подаваемых на дерево признаков
print("\n1.2 Исследование зависимости качества от количества признаков на дерево")

n_features = X.shape[1]
max_features_options = range(1, n_features + 1)
f1_scores_features = []
train_times_features = []

for n_feat in max_features_options:
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=optimal_depth_rf, 
                               max_features=n_feat, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_features.append(train_time)
    
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_features.append(f1)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(max_features_options, f1_scores_features, marker='s', linewidth=2, markersize=8)
plt.xlabel('Количество признаков на дерево')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от количества признаков на дерево в случайном лесу')
plt.grid(True)
plt.xticks(max_features_options)
plt.show()

# Определение оптимального количества признаков
optimal_features_rf = max_features_options[np.argmax(f1_scores_features)]
print(f"Оптимальное количество признаков на дерево: {optimal_features_rf}")
print(f"Максимальный F1-score: {max(f1_scores_features):.4f}")

# исследование качества модели от числа деревьев
print("\n1.3 Исследование зависимости качества и времени обучения от числа деревьев")

n_estimators_options = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
f1_scores_estimators = []
train_times_estimators = []

for n_est in n_estimators_options:
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=optimal_depth_rf, 
                               max_features=optimal_features_rf, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_estimators.append(train_time)
    
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_estimators.append(f1)

# построение графика качества
fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel('Количество деревьев')
ax1.set_ylabel('F1-score', color=color)
ax1.plot(n_estimators_options, f1_scores_estimators, marker='o', linewidth=2, markersize=8, color=color, label='F1-score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Время обучения (сек)', color=color)
ax2.plot(n_estimators_options, train_times_estimators, marker='s', linewidth=2, markersize=8, color=color, label='Время обучения')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Зависимость F1-score и времени обучения от количества деревьев в случайном лесу')
fig.tight_layout()
plt.show()

# Определение оптимального количества деревьев
optimal_estimators_rf = n_estimators_options[np.argmax(f1_scores_estimators)]
print(f"Оптимальное количество деревьев: {optimal_estimators_rf}")
print(f"Максимальный F1-score: {max(f1_scores_estimators):.4f}")
print(f"Время обучения для оптимальной модели: {train_times_estimators[np.argmax(f1_scores_estimators)]:.2f} сек")

# Обучение финальной модели случайного леса с оптимальными параметрами
rf_final = RandomForestClassifier(n_estimators=optimal_estimators_rf, 
                                 max_depth=optimal_depth_rf, 
                                 max_features=optimal_features_rf, 
                                 random_state=42)
start_time = time.time()
rf_final.fit(X_train, y_train)
rf_train_time = time.time() - start_time
y_pred_rf_final = rf_final.predict(X_test)

print(f"\nФинальная модель случайного леса:")
print(f"F1-score: {f1_score(y_test, y_pred_rf_final):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_final):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_final):.4f}")
print(f"Время обучения: {rf_train_time:.2f} сек")

# Важность признаков для случайного леса
feature_importances_rf = rf_final.feature_importances_
feature_names = X.columns.tolist()
indices_rf = np.argsort(feature_importances_rf)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков (Случайный лес)")
plt.bar(range(len(feature_importances_rf)), feature_importances_rf[indices_rf], align="center")
plt.xticks(range(len(feature_importances_rf)), [feature_names[i] for i in indices_rf], rotation=45)
plt.xlim([-1, len(feature_importances_rf)])
plt.tight_layout()
plt.show()

print("\nВажность признаков (Случайный лес):")
for i in indices_rf:
    print(f"{feature_names[i]}: {feature_importances_rf[i]:.4f}")

# Задача  XGBoost

print("\n" + "="*60)
print("ЗАДАЧА 2: XGBOOST")
print("="*60)

# Исследование гиперпараметров XGBoost

# Исследование зависимости от max_depth
print("\n2.1 Исследование зависимости качества от max_depth в XGBoost")

max_depths_xgb = range(1, 16)
f1_scores_xgb_depth = []
train_times_xgb_depth = []

for depth in max_depths_xgb:
    start_time = time.time()
    xgb_model = XGBClassifier(n_estimators=100, max_depth=depth, learning_rate=0.1, 
                             random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_xgb_depth.append(train_time)
    
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_xgb_depth.append(f1)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(max_depths_xgb, f1_scores_xgb_depth, marker='o', linewidth=2, markersize=8)
plt.xlabel('Максимальная глубина деревьев')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины деревьев в XGBoost')
plt.grid(True)
plt.xticks(max_depths_xgb)
plt.show()

optimal_depth_xgb = max_depths_xgb[np.argmax(f1_scores_xgb_depth)]
print(f"Оптимальная глубина деревьев для XGBoost: {optimal_depth_xgb}")
print(f"Максимальный F1-score: {max(f1_scores_xgb_depth):.4f}")

# Исследование зависимости от learning_rate
print("\n2.2 Исследование зависимости качества от learning_rate в XGBoost")

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
f1_scores_xgb_lr = []
train_times_xgb_lr = []

for lr in learning_rates:
    start_time = time.time()
    xgb_model = XGBClassifier(n_estimators=100, max_depth=optimal_depth_xgb, learning_rate=lr, 
                             random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_xgb_lr.append(train_time)
    
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_xgb_lr.append(f1)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(learning_rates, f1_scores_xgb_lr, marker='s', linewidth=2, markersize=8)
plt.xlabel('Learning Rate')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от learning_rate в XGBoost')
plt.grid(True)
plt.xscale('log')
plt.show()

optimal_lr_xgb = learning_rates[np.argmax(f1_scores_xgb_lr)]
print(f"Оптимальный learning_rate для XGBoost: {optimal_lr_xgb}")
print(f"Максимальный F1-score: {max(f1_scores_xgb_lr):.4f}")

# Исследование зависимости от subsample
print("\n2.3 Исследование зависимости качества от subsample в XGBoost")

subsample_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
f1_scores_xgb_subsample = []
train_times_xgb_subsample = []

for subsample in subsample_values:
    start_time = time.time()
    xgb_model = XGBClassifier(n_estimators=100, max_depth=optimal_depth_xgb, learning_rate=optimal_lr_xgb, 
                             subsample=subsample, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_xgb_subsample.append(train_time)
    
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_xgb_subsample.append(f1)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(subsample_values, f1_scores_xgb_subsample, marker='^', linewidth=2, markersize=8)
plt.xlabel('Subsample')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от subsample в XGBoost')
plt.grid(True)
plt.xticks(subsample_values)
plt.show()

optimal_subsample_xgb = subsample_values[np.argmax(f1_scores_xgb_subsample)]
print(f"Оптимальный subsample для XGBoost: {optimal_subsample_xgb}")
print(f"Максимальный F1-score: {max(f1_scores_xgb_subsample):.4f}")

#  Исследование зависимости от n_estimators с учетом времени обучения
print("\n2.4 Исследование зависимости качества и времени обучения от n_estimators в XGBoost")

n_estimators_xgb_options = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
f1_scores_xgb_estimators = []
train_times_xgb_estimators = []

for n_est in n_estimators_xgb_options:
    start_time = time.time()
    xgb_model = XGBClassifier(n_estimators=n_est, max_depth=optimal_depth_xgb, 
                             learning_rate=optimal_lr_xgb, subsample=optimal_subsample_xgb,
                             random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times_xgb_estimators.append(train_time)
    
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores_xgb_estimators.append(f1)

# Построение графика
fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel('Количество деревьев (n_estimators)')
ax1.set_ylabel('F1-score', color=color)
ax1.plot(n_estimators_xgb_options, f1_scores_xgb_estimators, marker='o', linewidth=2, markersize=8, color=color, label='F1-score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Время обучения (сек)', color=color)
ax2.plot(n_estimators_xgb_options, train_times_xgb_estimators, marker='s', linewidth=2, markersize=8, color=color, label='Время обучения')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Зависимость F1-score и времени обучения от количества деревьев в XGBoost')
fig.tight_layout()
plt.show()

optimal_estimators_xgb = n_estimators_xgb_options[np.argmax(f1_scores_xgb_estimators)]
print(f"Оптимальное количество деревьев для XGBoost: {optimal_estimators_xgb}")
print(f"Максимальный F1-score: {max(f1_scores_xgb_estimators):.4f}")
print(f"Время обучения для оптимальной модели: {train_times_xgb_estimators[np.argmax(f1_scores_xgb_estimators)]:.2f} сек")

# Обучение финальной модели XGBoost с оптимальными параметрами
xgb_final = XGBClassifier(n_estimators=optimal_estimators_xgb, 
                         max_depth=optimal_depth_xgb, 
                         learning_rate=optimal_lr_xgb,
                         subsample=optimal_subsample_xgb,
                         random_state=42, 
                         use_label_encoder=False, 
                         eval_metric='logloss')
start_time = time.time()
xgb_final.fit(X_train, y_train)
xgb_train_time = time.time() - start_time
y_pred_xgb_final = xgb_final.predict(X_test)

print(f"\nФинальная модель XGBoost:")
print(f"F1-score: {f1_score(y_test, y_pred_xgb_final):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_final):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb_final):.4f}")
print(f"Время обучения: {xgb_train_time:.2f} сек")

# Важность признаков для XGBoost
feature_importances_xgb = xgb_final.feature_importances_
indices_xgb = np.argsort(feature_importances_xgb)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков (XGBoost)")
plt.bar(range(len(feature_importances_xgb)), feature_importances_xgb[indices_xgb], align="center")
plt.xticks(range(len(feature_importances_xgb)), [feature_names[i] for i in indices_xgb], rotation=45)
plt.xlim([-1, len(feature_importances_xgb)])
plt.tight_layout()
plt.show()

print("\nВажность признаков (XGBoost):")
for i in indices_xgb:
    print(f"{feature_names[i]}: {feature_importances_xgb[i]:.4f}")

# Сравнение моделей
print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*60)

# Сравнение метрик
models_comparison = {
    'Модель': ['Случайный лес', 'XGBoost'],
    'F1-score': [f1_score(y_test, y_pred_rf_final), f1_score(y_test, y_pred_xgb_final)],
    'Accuracy': [accuracy_score(y_test, y_pred_rf_final), accuracy_score(y_test, y_pred_xgb_final)],
    'Precision': [precision_score(y_test, y_pred_rf_final), precision_score(y_test, y_pred_xgb_final)],
    'Recall': [recall_score(y_test, y_pred_rf_final), recall_score(y_test, y_pred_xgb_final)],
    'Время обучения (сек)': [rf_train_time, xgb_train_time]
}

comparison_df = pd.DataFrame(models_comparison)
print(comparison_df.to_string(index=False))

# Выводы
print("\n" + "="*60)
print("ВЫВОДЫ ПО ЛАБОРАТОРНОЙ РАБОТЕ")
print("="*60)

print("1. Случайный лес:")
print(f"   - Оптимальные параметры: глубина={optimal_depth_rf}, признаков на дерево={optimal_features_rf}, деревьев={optimal_estimators_rf}")
print(f"   - Лучший F1-score: {max(f1_scores_estimators):.4f}")
print(f"   - Время обучения: {rf_train_time:.2f} сек")

print("\n2. XGBoost:")
print(f"   - Оптимальные параметры: глубина={optimal_depth_xgb}, learning_rate={optimal_lr_xgb}, subsample={optimal_subsample_xgb}, деревьев={optimal_estimators_xgb}")
print(f"   - Лучший F1-score: {max(f1_scores_xgb_estimators):.4f}")
print(f"   - Время обучения: {xgb_train_time:.2f} сек")

print(f"\n3. Сравнение моделей:")
if f1_score(y_test, y_pred_rf_final) > f1_score(y_test, y_pred_xgb_final):
    better_model = "Случайный лес"
    quality_diff = f1_score(y_test, y_pred_rf_final) - f1_score(y_test, y_pred_xgb_final)
else:
    better_model = "XGBoost"
    quality_diff = f1_score(y_test, y_pred_xgb_final) - f1_score(y_test, y_pred_rf_final)

if rf_train_time < xgb_train_time:
    faster_model = "Случайный лес"
    time_diff = xgb_train_time - rf_train_time
else:
    faster_model = "XGBoost"
    time_diff = rf_train_time - xgb_train_time

print(f"   - Лучшее качество (F1-score): {better_model} (разница: {quality_diff:.4f})")
print(f"   - Быстрее обучается: {faster_model} (разница: {time_diff:.2f} сек)")

