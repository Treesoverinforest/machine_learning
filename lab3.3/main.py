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

# Удалить все строки, содержащие пропуски
initial_shape = df.shape
df_cleaned = df.dropna()
after_dropna_shape = df_cleaned.shape

print(f"\nРазмер до удаления пропусков: {initial_shape}")
print(f"Размер после удаления пропусков: {after_dropna_shape}")


# Оставляем только числовые столбцы + Sex и Embarked
columns_to_keep = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
columns_to_keep += ['Sex', 'Embarked']  # добавляем Sex и Embarked


df_filtered = df_cleaned[columns_to_keep]


if 'PassengerId' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns=['PassengerId'])

print(f"\nСтолбцы после фильтрации: {list(df_filtered.columns)}")

# Sex: male=0, female=1
df_filtered['Sex'] = df_filtered['Sex'].map({'male': 0, 'female': 1})

# Embarked: C=1, Q=2, S=3
df_filtered['Embarked'] = df_filtered['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

# Проверим, что перекодировка прошла успешно
print("\nПервые 5 строк после перекодировки:")
print(df_filtered.head())

# Потеря строк: после dropna
rows_lost = initial_shape[0] - after_dropna_shape[0]
percent_rows_lost = (rows_lost / initial_shape[0]) * 100

# Потеря столбцов: исходные столбцы минус оставшиеся (без PassengerId)
initial_columns = set(df.columns) - {'PassengerId'}  # исключаем PassengerId из подсчёта потерь
final_columns = set(df_filtered.columns)
columns_lost = initial_columns - final_columns
percent_columns_lost = (len(columns_lost) / len(initial_columns)) * 100

print(f"\nПроцент потерянных строк: {percent_rows_lost:.2f}%")
print(f"Процент потерянных столбцов: {percent_columns_lost:.2f}%")

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

X_train_no_embarked = X_train.drop(columns=['Embarked'])
X_test_no_embarked = X_test.drop(columns=['Embarked'])

model_no_embarked = LogisticRegression(random_state=42, max_iter=1000)
model_no_embarked.fit(X_train_no_embarked, y_train)

y_pred_no_embarked = model_no_embarked.predict(X_test_no_embarked)
accuracy_no_embarked = accuracy_score(y_test, y_pred_no_embarked)

print(f"\nТочность модели БЕЗ признака Embarked: {accuracy_no_embarked:.4f} ({accuracy_no_embarked*100:.2f}%)")
print(f"Разница в точности: {accuracy - accuracy_no_embarked:.4f}")

if accuracy > accuracy_no_embarked:
    print("Признак Embarked УЛУЧШАЕТ точность модели.")
elif accuracy < accuracy_no_embarked:
    print("Признак Embarked УХУДШАЕТ точность модели.")
else:
    print("Признак Embarked не влияет на точность модели.")
    

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#  Метрики и визуализации для логистической регрессии ---

print("=== Часть 3.1: Оценка модели логистической регрессии ===\n")

# 1. Precision, Recall, F1
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred)

print(f"Precision: {precision_lr:.4f}")
print(f"Recall:    {recall_lr:.4f}")
print(f"F1-score:  {f1_lr:.4f}\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 2. Матрица ошибок (Confusion Matrix) + тепловая карта
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Погиб (0)', 'Выжил (1)'],
            yticklabels=['Погиб (0)', 'Выжил (1)'])
plt.title('Матрица ошибок (Confusion Matrix)')
plt.xlabel('Предсказанные значения')
plt.ylabel('Истинные значения')
plt.show()

# 3. PR-кривая (Precision-Recall Curve)
# Получаем вероятности для класса "1" (выжил)
y_proba_lr = model.predict_proba(X_test)[:, 1]

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_lr)
ap_lr = average_precision_score(y_test, y_proba_lr)  # Average Precision

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'Logistic Regression (AP = {ap_lr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend()
plt.grid(True)
plt.show()

print(f"Average Precision (AP): {ap_lr:.4f}\n")

# 4. ROC-кривая
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc_lr:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"AUC-ROC: {roc_auc_lr:.4f}\n")

# === Вывод по модели логистической регрессии ===
print("="*60)
print("ВЫВОД ПО МОДЕЛИ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ:")
print("="*60)
print(f"- Точность (accuracy):       {accuracy:.4f}")
print(f"- Precision:                 {precision_lr:.4f}")
print(f"- Recall:                    {recall_lr:.4f}")
print(f"- F1-score:                  {f1_lr:.4f}")
print(f"- AUC-ROC:                   {roc_auc_lr:.4f}")
print(f"- Average Precision (AP):    {ap_lr:.4f}")

if roc_auc_lr > 0.85:
    print(" Модель имеет ОТЛИЧНОЕ качество.")
elif roc_auc_lr > 0.75:
    print("Модель имеет ХОРОШЕЕ качество.")
else:
    print("Качество модели УДОВЛЕТВОРИТЕЛЬНОЕ или СЛАБОЕ.")

print("\nPR-кривая особенно важна при несбалансированных данных (в Titanic класс 'выжил' — меньшинство).")
print("Матрица ошибок показывает, на каком классе модель ошибается чаще.")
print("="*60)

# Обучение SVM и KNN + сравнение моделей ---

print("\n\n=== Часть 3.2: Сравнение с SVM и KNN ===")

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Словарь для хранения результатов всех моделей
results = {}

# Добавляем логистическую регрессию в результаты
results['Logistic Regression'] = {
    'accuracy': accuracy,
    'precision': precision_lr,
    'recall': recall_lr,
    'f1': f1_lr,
    'roc_auc': roc_auc_lr,
    'ap': ap_lr
}

# Модель 1 SVM
print("\n[Модель: SVM]")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)[:, 1]

precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
ap_svm = average_precision_score(y_test, y_proba_svm)

results['SVM'] = {
    'accuracy': svm_model.score(X_test, y_test),
    'precision': precision_svm,
    'recall': recall_svm,
    'f1': f1_svm,
    'roc_auc': roc_auc_svm,
    'ap': ap_svm
}

print(f"Accuracy:  {results['SVM']['accuracy']:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall:    {recall_svm:.4f}")
print(f"F1-score:  {f1_svm:.4f}")
print(f"AUC-ROC:   {roc_auc_svm:.4f}")
print(f"AP:        {ap_svm:.4f}")

# Модель 2KNN 
print("\n[Модель: KNN]")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_proba_knn = knn_model.predict_proba(X_test)[:, 1]

precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
ap_knn = average_precision_score(y_test, y_proba_knn)

results['KNN'] = {
    'accuracy': knn_model.score(X_test, y_test),
    'precision': precision_knn,
    'recall': recall_knn,
    'f1': f1_knn,
    'roc_auc': roc_auc_knn,
    'ap': ap_knn
}

print(f"Accuracy:  {results['KNN']['accuracy']:.4f}")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall:    {recall_knn:.4f}")
print(f"F1-score:  {f1_knn:.4f}")
print(f"AUC-ROC:   {roc_auc_knn:.4f}")
print(f"AP:        {ap_knn:.4f}")

# Сравнение всех моделей
print("\n\n" + "="*70)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*70)

import pandas as pd
comparison_df = pd.DataFrame(results).T.round(4)
print(comparison_df)

# Определяем лучшие модели по ключевым метрикам
best_f1 = comparison_df['f1'].idxmax()
best_auc = comparison_df['roc_auc'].idxmax()
best_ap = comparison_df['ap'].idxmax()

print(f"\nЛучшая модель по F1-score:     {best_f1} ({comparison_df.loc[best_f1, 'f1']:.4f})")
print(f" Лучшая модель по AUC-ROC:     {best_auc} ({comparison_df.loc[best_auc, 'roc_auc']:.4f})")
print(f"Лучшая модель по AP:          {best_ap} ({comparison_df.loc[best_ap, 'ap']:.4f})")


#Совмещённая ROC-кривая для всех трёх моделей
plt.figure(figsize=(10, 8))

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})', lw=2)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.4f})', lw=2)
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.4f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Сравнение ROC-кривых моделей')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()