Instrucciones para Usar los Modelos 

Clonar el Repositorio
Para comenzar, el primer paso es clonar el repositorio Git donde se encuentran los archivos del proyecto. Esto se puede hacer desde la terminal utilizando el siguiente comando:


git clone <URL_DEL_REPOSITORIO>
Reemplaza <URL_DEL_REPOSITORIO> con el enlace del repositorio Git.

Instalar las Dependencias
Este proyecto utiliza varias bibliotecas de Python para la manipulación de datos, el preprocesamiento y el entrenamiento de los modelos. Antes de ejecutar los scripts, es necesario instalar las dependencias utilizando pip. Para hacerlo, asegúrate de tener un entorno virtual o de estar en un entorno controlado, y luego ejecuta:


pip install -r requirements.txt
Esto instalará todas las bibliotecas necesarias como Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost, entre otras.

Cargar los Datos CSV
Asegúrate de que el archivo CSV data_tfm_cleaned.csv o WhatsgoodlyData-6.csv esté en la carpeta del proyecto o disponible para su carga. Para cargar el archivo en los scripts de modelado, usa el siguiente código en Python:


import pandas as pd
# Cargar el archivo CSV
df = pd.read_csv('data_tfm_cleaned.csv')
# Ver los primeros datos para asegurarse de que se cargó correctamente
print(df.head())

Preprocesamiento de los Datos
El preprocesamiento es un paso clave antes de entrenar los modelos. En los scripts del repositorio, ya se implementa el preprocesamiento básico de los datos, como la imputación de valores faltantes y la codificación de variables categóricas:


# Imputar valores faltantes en la columna objetivo
df['Answer'].fillna('Desconocido', inplace=True)
# Imputar valores faltantes en columnas numéricas
df.fillna(df.mean(), inplace=True)
Entrenar los Modelos
Para entrenar los diferentes modelos, puedes ejecutar los scripts correspondientes. Por ejemplo, para entrenar un modelo de Árboles Aleatorios:

python random_forest_model.py
El script mostrará las métricas de evaluación, como la matriz de confusión y el reporte de clasificación.

Validación de Resultados


from sklearn.metrics import ConfusionMatrixDisplay
# Matriz de Confusión
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=label_encoder.classes_).plot()
plt.show()
Ajuste de Hiperparámetros
Si deseas mejorar los modelos, puedes ajustar los hiperparámetros utilizando GridSearchCV:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
Revisión del Código Fuente
El código fuente de los modelos está disponible en el repositorio. Para cualquier ajuste o personalización, revisa los archivos .py correspondientes.

Comparación de Modelos
Para comparar el desempeño de los distintos modelos, se incluye un cuadro comparativo con las métricas clave de cada uno. Estos resultados pueden ser reproducidos ejecutando los scripts del repositorio.
