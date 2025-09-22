from sklearn.datasets import load_iris
import pandas as pd

# Загрузка датасета
iris = load_iris()

# Преобразование в DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Вывод имен классов и целевой переменной
print("Имена сортов:", iris.target_names)
print("Целевая переменная (первые 10 значений):", iris.target[:10])

import matplotlib.pyplot as plt

# Создаем фигуру с двумя подграфиками
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Цвета для каждого класса
colors = ['red', 'green', 'blue']
labels = iris.target_names

# График 1: sepal length vs sepal width
for i, color in enumerate(colors):
    mask = df['target'] == i
    axes[0].scatter(df.loc[mask, 'sepal length (cm)'],
                    df.loc[mask, 'sepal width (cm)'],
                    c=color, label=labels[i], alpha=0.7)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].set_title('Sepal Length vs Sepal Width')
axes[0].legend()
axes[0].grid(True)

# График 2: petal length vs petal width
for i, color in enumerate(colors):
    mask = df['target'] == i
    axes[1].scatter(df.loc[mask, 'petal length (cm)'],
                    df.loc[mask, 'petal width (cm)'],
                    c=color, label=labels[i], alpha=0.7)
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].set_title('Petal Length vs Petal Width')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

import seaborn as sns

# Добавим столбец с именами классов для наглядности
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Построение pairplot
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle('Pairplot Iris Dataset', y=1.02)
plt.show()

# Датасет 1: setosa (0) и versicolor (1)
df1 = df[df['target'].isin([0, 1])].copy()
X1 = df1.drop(['target', 'species'], axis=1)
y1 = df1['target']

# Датасет 2: versicolor (1) и virginica (2)
df2 = df[df['target'].isin([1, 2])].copy()
# Перекодируем метки: versicolor=0, virginica=1 для удобства
df2['target_binary'] = df2['target'].map({1: 0, 2: 1})
X2 = df2.drop(['target', 'species', 'target_binary'], axis=1)
y2 = df2['target_binary']

print(f"Датасет 1 (setosa vs versicolor): {X1.shape[0]} образцов")
print(f"Датасет 2 (versicolor vs virginica): {X2.shape[0]} образцов")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_and_evaluate(X, y, dataset_name):
    print(f"\n=== {dataset_name} ===")
    
    # 4. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. Создание модели
    clf = LogisticRegression(random_state=0, max_iter=200)
    
    # 6. Обучение модели
    clf.fit(X_train, y_train)
    
    # 7. Предсказание
    y_pred = clf.predict(X_test)
    
    # 8. Точность модели
    accuracy = clf.score(X_test, y_test)
    print(f"Точность (accuracy) на тестовой выборке: {accuracy:.4f}")
    
    return clf, X_train, X_test, y_train, y_test

# Обучение для первого датасета
model1, X1_train, X1_test, y1_train, y1_test = train_and_evaluate(X1, y1, "Setosa vs Versicolor")

# Обучение для второго датасета
model2, X2_train, X2_test, y2_train, y2_test = train_and_evaluate(X2, y2, "Versicolor vs Virginica")

from sklearn.datasets import make_classification

# Генерация данных
X_gen, y_gen = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1
)

# Визуализация сгенерированного датасета
plt.figure(figsize=(8, 6))
plt.scatter(X_gen[:, 0], X_gen[:, 1], c=y_gen, cmap='coolwarm', alpha=0.7)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Сгенерированный датасет для бинарной классификации')
plt.grid(True)
plt.show()

# Проведем классификацию (пункты 5-8)
X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(
    X_gen, y_gen, test_size=0.3, random_state=42, stratify=y_gen
)

clf_gen = LogisticRegression(random_state=0, max_iter=200)
clf_gen.fit(X_train_gen, y_train_gen)
accuracy_gen = clf_gen.score(X_test_gen, y_test_gen)

print(f"\n=== Сгенерированный датасет ===")
print(f"Точность (accuracy) на тестовой выборке: {accuracy_gen:.4f}")