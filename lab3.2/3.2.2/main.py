#Многоклассовая логистическая регрессия на Iris

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
iris = load_iris()
X = iris.data[:, 2:4]  # petal length, petal width
y = iris.target

feature_names = iris.feature_names[2:4]
target_names = iris.target_names

print("Признаки:", feature_names)
print("Классы:", target_names)

# Масштабирование данных 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение модели
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

# Создание сетки для визуализации
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Предсказание для каждой точки сетки
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Визуализация
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel(f'{feature_names[0]} (scaled)')
plt.ylabel(f'{feature_names[1]} (scaled)')
plt.title('Многоклассовая логистическая регрессия на Iris (petal length vs petal width)')
plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
plt.grid(True)
plt.show()

# Точность модели
accuracy = model.score(X_scaled, y)
print(f"\nТочность модели на обучающих данных: {accuracy:.4f} ({accuracy*100:.2f}%)")