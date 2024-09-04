import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

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

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarizar las clases DESPUÉS de la división de los datos
y_train_bin = label_binarize(y_train, classes=y.unique())
y_test_bin = label_binarize(y_test, classes=y.unique())

n_classes = y_train_bin.shape[1]  # Definir el número de clases

# Modelo Random Forest
rf_model = RandomForestClassifier(
    n_estimators=10,         
    max_depth=3,             
    min_samples_split=15,    
    min_samples_leaf=10,     
    max_features='log2',     
    bootstrap=True,
    random_state=42
)

# Convertir a One-vs-Rest para clasificación multiclase
ovr_rf_model = OneVsRestClassifier(rf_model)

# Entrenar el modelo con el conjunto de entrenamiento
ovr_rf_model.fit(X_train, y_train_bin)

# Predecir probabilidades en el conjunto de prueba
y_score = ovr_rf_model.predict_proba(X_test)

# Convertir las predicciones numéricas a las clases originales
y_pred_indices = y_score.argmax(axis=1)
y_pred = np.array([y.unique()[index] for index in y_pred_indices])

# Matriz de Confusión
print("Matriz de Confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualización de la Matriz de Confusión
plt.figure(figsize=(6, 6))  # Configura el tamaño del gráfico
ConfusionMatrixDisplay(conf_matrix, display_labels=ovr_rf_model.classes_).plot()
plt.title('Matriz de Confusión - Random Forest')
plt.show()

# Reporte de Clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Calcular el AUC para cada clase y el promedio macro
roc_auc = {}
for i in range(n_classes):
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

print("AUC para cada clase:")
for i in roc_auc:
    print(f'Clase {y.unique()[i]}: AUC = {roc_auc[i]:.2f}')

# Promedio macro de AUC
auc_macro = np.mean(list(roc_auc.values()))
print(f'Promedio Macro AUC: {auc_macro:.2f}')

# Curvas ROC para cada clase
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f'ROC Clase {y.unique()[i]} (AUC = {roc_auc[i]:.2f})')

# Graficar la línea diagonal (azar)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC - Clasificación Multiclase (Random Forest)')
plt.legend(loc="lower right")
plt.show()
