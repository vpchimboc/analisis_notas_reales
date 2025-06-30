
# ==============================================
# Análisis Exploratorio de Datos - Ejemplo Real (Corregido)
# ==============================================

# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ----------------------------------------------
# 1. Cargar datos
# ----------------------------------------------
# Teoría: Se inicia cargando el archivo maestro de notas
df_real = pd.read_excel('MAESTRO_DE_NOTAS.xlsx', sheet_name='rpt_maestro_notas')
print(df_real.head())

# ----------------------------------------------
# 2. Agrupamiento con Pandas
# ----------------------------------------------
# Teoría: Agrupa por asignatura y calcula promedio de notas
promedios = df_real.groupby('Asignatura')['Nota final'].mean()
print(promedios)

# ----------------------------------------------
# 3. Histograma de Notas Finales
# ----------------------------------------------
# Teoría: Visualiza la distribución de notas
plt.hist(df_real['Nota final'].dropna(), bins=10)
plt.title('Distribución de Notas Finales')
plt.xlabel('Nota')
plt.ylabel('Frecuencia')
plt.show()

# ----------------------------------------------
# 4. Boxplot por Asignatura
# ----------------------------------------------
# Teoría: Muestra dispersión de notas y posibles outliers
plt.figure(figsize=(12, 6))
sns.boxplot(x='Asignatura', y='Nota final', data=df_real)
plt.xticks(rotation=90)
plt.title('Boxplot de Notas por Asignatura')
plt.show()

# ----------------------------------------------
# 5. Matriz de Correlación
# ----------------------------------------------
# Teoría: Explora relación entre asistencia y nota final
corr = df_real[['Nota final', 'Asistencia']].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# ----------------------------------------------
# 6. Clustering de Estudiantes (Corregido)
# ----------------------------------------------
# Teoría: Agrupa estudiantes por nota y asistencia
X = df_real[['Nota final', 'Asistencia']].dropna()
kmeans = KMeans(n_clusters=3)
df_real.loc[X.index, 'Cluster'] = kmeans.fit_predict(X)

print(df_real[['Estudiante', 'Nota final', 'Asistencia', 'Cluster']].head())

# ----------------------------------------------
# 7. Procesamiento de Texto (Simulado)
# ----------------------------------------------
# Teoría: Analiza un comentario de retroalimentación (opcional)
from textblob import TextBlob
comentario = TextBlob("El estudiante mostró avance significativo.")
print(comentario.sentiment)
