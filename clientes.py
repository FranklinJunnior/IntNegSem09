import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

# Cargar el archivo Excel
df_clientes = pd.read_excel('D:/InteligenciaNegocios/Semana09/BI_Clientes09-1.xlsx')

# Ver las primeras filas del DataFrame
print(df_clientes.head())

# Seleccionar las columnas relevantes para el modelo
X_clientes = df_clientes[['EnglishOccupation', 'SpanishOccupation', 'FrenchOccupation', 
                          'HouseOwnerFlag', 'NumberCarsOwned', 'CommuteDistance', 
                          'Region', 'Age']]

y_clientes = df_clientes['BikeBuyer']  # Variable objetivo

# Convertir columnas categóricas a variables dummy
X_clientes = pd.get_dummies(X_clientes, columns=['EnglishOccupation', 'SpanishOccupation', 
                                                 'FrenchOccupation', 'CommuteDistance', 'Region'])

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_clientes, y_clientes, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de árbol de decisiones
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualizar el árbol de decisiones
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['No', 'Yes'])
plt.show()

# Evaluar el modelo en el conjunto de prueba
accuracy = clf.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
