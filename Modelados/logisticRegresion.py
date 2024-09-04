import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Cargar los datos
df = pd.read_csv('data_tfm_preprocesado.csv')

# Llenar valores NaN en 'Answer'
df['Answer'].fillna('Desconocido', inplace=True)

# Asegurar que no queden valores NaN en las columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Selección de variables
features = ['Facebook', 'Instagram', 'Snapchat', 'Twitter']
X = df[features]
y = df['Answer']

# Convertir las etiquetas (y) a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarizar las etiquetas para calcular las curvas ROC/AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

# Modelo de Regresión Logística
logreg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logreg_model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred_logreg = logreg_model.predict(X_test)

# Evaluación del modelo
print("Matriz de Confusión:")
conf_matrix = confusion_matrix(y_test, y_pred_logreg)
print(conf_matrix)

# Visualización de la Matriz de Confusión
ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot()
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()

# Reporte de Clasificación del modelo de regresión logística
print("Reporte de Clasificación:")
target_names = label_encoder.classes_.astype(str)  # Asegurarse de que las clases son cadenas
print(classification_report(y_test, y_pred_logreg, target_names=target_names))

# Calcular las curvas ROC y AUC para cada clase
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], logreg_model.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotear las curvas ROC
plt.figure()
for i in range(len(np.unique(y))):
    plt.plot(fpr[i], tpr[i], label=f'ROC clase {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC - Regresión Logística')
plt.legend(loc="lower right")
plt.show()

# Validación cruzada para verificar la generalización del modelo
scores = cross_val_score(logreg_model, X, y, cv=10, scoring='accuracy')

# Mostrar la precisión media obtenida en la validación cruzada
print(f"Precisión media en validación cruzada: {np.mean(scores)}")
