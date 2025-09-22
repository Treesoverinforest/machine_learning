# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('Titanic.csv')

print("Первые 5 строк датасета:")
print(df.head())

# Задание 1.1: Удалить все строки, содержащие пропуски
initial_shape = df.shape
df_cleaned = df.dropna()
after_dropna_shape = df_cleaned.shape

print(f"\nРазмер до удаления пропусков: {initial_shape}")
print(f"Размер после удаления пропусков: {after_dropna_shape}")

# Оставляем только числовые столбцы + Sex и Embarked
columns_to_keep = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
columns_to_keep += ['Sex', 'Embarked']  # добавляем Sex и Embarked

# Удаляем остальные столбцы
df_filtered = df_cleaned[columns_to_keep]

if 'PassengerId' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns=['PassengerId'])

print(f"\nСтолбцы после фильтрации: {list(df_filtered.columns)}")

 #Перекодировать Sex и Embarked в числовые значения
# Sex: male=0, female=1
df_filtered['Sex'] = df_filtered['Sex'].map({'male': 0, 'female': 1})

# Embarked: C=1, Q=2, S=3
df_filtered['Embarked'] = df_filtered['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

# Проверим, что перекодировка прошла успешно
print("\nПервые 5 строк после перекодировки:")
print(df_filtered.head())

#Вычислить процент потерянных данных после пп. 1.1–1.2

rows_lost = initial_shape[0] - after_dropna_shape[0]
percent_rows_lost = (rows_lost / initial_shape[0]) * 100

# Потеря столбцов: исходные столбцы минус оставшиеся (без PassengerId)
initial_columns = set(df.columns) - {'PassengerId'}  # исключаем PassengerId из подсчёта потерь
final_columns = set(df_filtered.columns)
columns_lost = initial_columns - final_columns
percent_columns_lost = (len(columns_lost) / len(initial_columns)) * 100

print(f"\nПроцент потерянных строк: {percent_rows_lost:.2f}%")
print(f"Процент потерянных столбцов: {percent_columns_lost:.2f}%")

# Подготовка данных для обучения
X = df_filtered.drop(columns=['Survived'])  # признаки
y = df_filtered['Survived']                 # целевая переменная


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")


model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели логистической регрессии: {accuracy:.4f} ({accuracy*100:.2f}%)")

#Оценить влияние признака Embarked на точность модели
# Обучим модель без признака Embarked
X_train_no_embarked = X_train.drop(columns=['Embarked'])
X_test_no_embarked = X_test.drop(columns=['Embarked'])

model_no_embarked = LogisticRegression(random_state=42, max_iter=1000)
model_no_embarked.fit(X_train_no_embarked, y_train)

y_pred_no_embarked = model_no_embarked.predict(X_test_no_embarked)
accuracy_no_embarked = accuracy_score(y_test, y_pred_no_embarked)

print(f"\nТочность модели БЕЗ признака Embarked: {accuracy_no_embarked:.4f} ({accuracy_no_embarked*100:.2f}%)")
print(f"Разница в точности: {accuracy - accuracy_no_embarked:.4f}")

if accuracy > accuracy_no_embarked:
    print(" Признак Embarked УЛУЧШАЕТ точность модели.")
elif accuracy < accuracy_no_embarked:
    print("Признак Embarked УХУДШАЕТ точность модели.")
else:
    print("Признак Embarked не влияет на точность модели.")