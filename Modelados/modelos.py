import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Cargar los datos preprocesados
df = pd.read_csv('data_tfm_preprocesado.csv')

# Asegurar que no queden valores faltantes en 'Answer'
df['Answer'].fillna('Desconocido', inplace=True)

# Selección de variables
features = ['Facebook', 'Instagram', 'Snapchat', 'Twitter']
X = df[features]
y = df['Answer']

# Convertir la variable 'Answer' en categorías numéricas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Asegurarse de que las clases sean cadenas de texto
target_names = label_encoder.classes_.astype(str)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. SVM (Support Vector Machine)
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1, decision_function_shape='ovr')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("Matriz de Confusión - SVM:")
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print(conf_matrix_svm)
ConfusionMatrixDisplay(conf_matrix_svm, display_labels=target_names).plot()
plt.title('Matriz de Confusión - SVM')
plt.show()

print("Reporte de Clasificación - SVM:")
print(classification_report(y_test, y_pred_svm, target_names=target_names))

# 2. Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("Matriz de Confusión - Naive Bayes:")
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print(conf_matrix_nb)
ConfusionMatrixDisplay(conf_matrix_nb, display_labels=target_names).plot()
plt.title('Matriz de Confusión - Naive Bayes')
plt.show()

print("Reporte de Clasificación - Naive Bayes:")
print(classification_report(y_test, y_pred_nb, target_names=target_names))

# 3. K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("Matriz de Confusión - KNN:")
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print(conf_matrix_knn)
ConfusionMatrixDisplay(conf_matrix_knn, display_labels=target_names).plot()
plt.title('Matriz de Confusión - KNN')
plt.show()

print("Reporte de Clasificación - KNN:")
print(classification_report(y_test, y_pred_knn, target_names=target_names))

# 4. Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("Matriz de Confusión - Gradient Boosting:")
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
print(conf_matrix_gb)
ConfusionMatrixDisplay(conf_matrix_gb, display_labels=target_names).plot()
plt.title('Matriz de Confusión - Gradient Boosting')
plt.show()

print("Reporte de Clasificación - Gradient Boosting:")
print(classification_report(y_test, y_pred_gb, target_names=target_names))

# 5. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Matriz de Confusión - Decision Tree:")
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print(conf_matrix_dt)
ConfusionMatrixDisplay(conf_matrix_dt, display_labels=target_names).plot()
plt.title('Matriz de Confusión - Decision Tree')
plt.show()

print("Reporte de Clasificación - Decision Tree:")
print(classification_report(y_test, y_pred_dt, target_names=target_names))

# 6. Perceptrón Multicapa (MLP)
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

print("Matriz de Confusión - MLP (Perceptrón Multicapa):")
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
print(conf_matrix_mlp)
ConfusionMatrixDisplay(conf_matrix_mlp, display_labels=target_names).plot()
plt.title('Matriz de Confusión - MLP (Perceptrón Multicapa)')
plt.show()

print("Reporte de Clasificación - MLP (Perceptrón Multicapa):")
print(classification_report(y_test, y_pred_mlp, target_names=target_names))
