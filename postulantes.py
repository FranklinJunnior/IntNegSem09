import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Cargar el archivo Excel
archivo_excel = 'D:/InteligenciaNegocios/Semana09/BI_Postulantes09-1.xlsx'
df_postulantes = pd.read_excel(archivo_excel)

# Ver las primeras filas del DataFrame
print(df_postulantes.head())

# Selecciona las columnas relevantes para el clustering
X_postulantes = df_postulantes[['Apertura Nuevos Conoc.', 'Nivel Organización', 
                                'Participación Grupo Social', 'Grado Empatía', 
                                'Grado Nerviosismo', 'Dependencia Internet']]

# Aplicar K-means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df_postulantes['Cluster'] = kmeans.fit_predict(X_postulantes)

# Crear histogramas cruzando clusters con especialidades
plt.figure(figsize=(10, 6))
especialidades = df_postulantes['Nom_Especialidad'].unique()

# Crear un histograma por especialidad
for especialidad in especialidades:
    subset = df_postulantes[df_postulantes['Nom_Especialidad'] == especialidad]
    sns.histplot(subset['Cluster'], bins=range(4), label=especialidad, kde=False, stat="count", alpha=0.7)

# Personalizar el gráfico
plt.xlabel('Cluster')
plt.ylabel('Frecuencia')
plt.legend(title='Especialidad', loc='upper right')
plt.title('Histogramas de Clusters por Especialidad')
plt.show()
