import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from xgboost import XGBClassifier
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

# Modelo XGBoost básico
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), eval_metric='mlogloss')

# Entrenar el modelo básico
xgb_model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred_xgb = xgb_model.predict(X_test)

# Evaluación del modelo básico
print("Matriz de Confusión (Modelo Básico):")
conf_matrix = confusion_matrix(y_test, y_pred_xgb)
print(conf_matrix)

# Visualización de la Matriz de Confusión
ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot()
plt.title('Matriz de Confusión - XGBoost (Básico)')
plt.show()

# Reporte de Clasificación del modelo básico
print("Reporte de Clasificación (Modelo Básico):")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))

# Definir el grid de hiperparámetros para optimización
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Configurar el modelo con GridSearchCV para optimización
grid_search = GridSearchCV(estimator=XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), eval_metric='mlogloss'),
                           param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Entrenar el modelo con GridSearchCV
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

# Usar el mejor modelo para hacer predicciones
best_xgb_model = grid_search.best_estimator_
y_pred_best_xgb = best_xgb_model.predict(X_test)

# Evaluación del modelo optimizado
print("Matriz de Confusión (Modelo Optimizado):")
conf_matrix_opt = confusion_matrix(y_test, y_pred_best_xgb)
print(conf_matrix_opt)

# Visualización de la Matriz de Confusión del modelo optimizado
ConfusionMatrixDisplay(conf_matrix_opt, display_labels=label_encoder.classes_).plot()
plt.title('Matriz de Confusión - XGBoost (Optimizado)')
plt.show()

# Reporte de Clasificación del modelo optimizado
print("Reporte de Clasificación (Modelo Optimizado):")
print(classification_report(y_test, y_pred_best_xgb, target_names=label_encoder.classes_))

# Binarizar solo las etiquetas del conjunto de prueba para calcular las curvas ROC/AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

# Curvas ROC y AUC para cada clase
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], best_xgb_model.predict_proba(X_test)[:, i])
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
plt.title('Curvas ROC - XGBoost Optimizado')
plt.legend(loc="lower right")
plt.show()

# Validación cruzada para verificar la generalización del modelo
scores = cross_val_score(best_xgb_model, X, y, cv=10, scoring='accuracy')

# Mostrar la precisión media obtenida en la validación cruzada
print(f"Precisión media en validación cruzada: {np.mean(scores)}")
