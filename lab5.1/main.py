import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import graphviz
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('diabetes.csv')

# Предварительный анализ данных
print("Первые 5 строк датасета:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())
print("\nОписательная статистика:")
print(df.describe())

# Проверка на пропущенные значения (в этом датасете 0 может означать отсутствие данных для некоторых признаков)
# Заменим нулевые значения в некоторых столбцах на медиану (кроме 'Pregnancies' и 'Outcome')
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, df[col][df[col] != 0].median())

# Разделение на признаки и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабирование данных для логистической регрессии
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Логистическая регрессия и решающее дерево со стандартными настройками

# Логистическая регрессия
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Решающее дерево
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Функция для вывода метрик
def print_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nМетрики для {model_name}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    return acc, prec, rec, f1

# Вывод метрик
metrics_lr = print_metrics(y_test, y_pred_lr, "Логистическая регрессия")
metrics_dt = print_metrics(y_test, y_pred_dt, "Решающее дерево")

# Вывод полного отчета классификации
print("\nОтчет классификации для логистической регрессии:")
print(classification_report(y_test, y_pred_lr))

print("\nОтчет классификации для решающего дерева:")
print(classification_report(y_test, y_pred_dt))

# Вывод по задаче 1
print("\n" + "="*60)
print("ВЫВОД ПО ЗАДАЧЕ 1:")
if metrics_lr[3] > metrics_dt[3]:  # Сравнение по F1-score
    print("Логистическая регрессия показала лучший F1-score и, следовательно, более сбалансированный результат.")
    print("Она лучше подходит для данного датасета, особенно если важна стабильность и обобщение.")
else:
    print("Решающее дерево показало лучший F1-score и, следовательно, более сбалансированный результат.")
    print("Оно лучше подходит для данного датасета, особенно если важна интерпретируемость.")

print("Примечание: Также стоит учитывать, что дерево может переобучаться, а логистическая регрессия — более устойчивая модель.")

#Исследование зависимости метрики от глубины дерева


depths = range(1, 21)
f1_scores_train = []
f1_scores_test = []

for depth in depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train, y_train)
    
    y_pred_train = dt_temp.predict(X_train)
    y_pred_test = dt_temp.predict(X_test)
    
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    
    f1_scores_train.append(f1_train)
    f1_scores_test.append(f1_test)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(depths, f1_scores_train, label='F1-score (обучающая выборка)', marker='o')
plt.plot(depths, f1_scores_test, label='F1-score (тестовая выборка)', marker='s')
plt.xlabel('Глубина дерева')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины решающего дерева')
plt.legend()
plt.grid(True)
plt.xticks(depths)
plt.show()

# Определение оптимальной глубины (по максимальному F1-score на тестовой выборке)
optimal_depth = depths[np.argmax(f1_scores_test)]
print(f"\nОптимальная глубина дерева: {optimal_depth}")
print(f"Максимальный F1-score на тестовой выборке: {max(f1_scores_test):.4f}")

#  Модель с оптимальной глубиной

# Обучение модели с оптимальной глубиной
dt_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
dt_optimal.fit(X_train, y_train)
y_pred_optimal = dt_optimal.predict(X_test)
y_proba_optimal = dt_optimal.predict_proba(X_test)[:, 1]

#Визуализация дерева
feature_names = X.columns.tolist()
class_names = ['No Diabetes', 'Diabetes']

dot_data = export_graphviz(dt_optimal, 
                           out_file=None,
                           feature_names=feature_names,
                           class_names=class_names,
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("diabetes_decision_tree", format='png', cleanup=True)
print("\nДерево решений сохранено в файл 'diabetes_decision_tree.png'")

Важность признаков
feature_importances = dt_optimal.feature_importances_
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# Создание DataFrame для важности признаков
importance_df = pd.DataFrame({
    'feature': [feature_names[i] for i in indices],
    'importance': feature_importances[indices]
})
print("\nВажность признаков:")
print(importance_df)

# 3.3 ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_proba_optimal)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#PR-кривая (Precision-Recall)
precision, recall, _ = precision_recall_curve(y_test, y_proba_optimal)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Дополнительно: Матрица ошибок для оптимальной модели
cm = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Матрица ошибок для оптимального дерева решений')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.show()
