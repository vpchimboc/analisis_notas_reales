
# ==============================================
# Análisis Exploratorio de Datos - Streamlit App
# ==============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from textblob import TextBlob

st.title("📊 Análisis Exploratorio de Datos - Notas Reales")

# Subir archivo
uploaded_file = st.file_uploader("📂 Sube tu archivo MAESTRO_DE_NOTAS.xlsx", type=['xlsx'])

if uploaded_file is not None:
    df_real = pd.read_excel(uploaded_file, sheet_name='rpt_maestro_notas')
    print(df_real['Identificacion']=="105739601")
    st.write(df_real['Identificacion']=="105739601")
    st.write("### Vista de los datos:", df_real.head())

    # Agrupamiento
    st.write("### Promedio por Asignatura")
    promedios = df_real.groupby('Asignatura')['Nota final'].mean()
    st.dataframe(promedios)

    # Histograma
    st.write("### Histograma de Notas Finales")
    fig1, ax1 = plt.subplots()
    ax1.hist(df_real['Nota final'].dropna(), bins=10)
    ax1.set_title('Distribución de Notas Finales')
    ax1.set_xlabel('Nota')
    ax1.set_ylabel('Frecuencia')
    st.pyplot(fig1)

    # Boxplot
    st.write("### Boxplot por Asignatura")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Asignatura', y='Nota final', data=df_real, ax=ax2)
    plt.xticks(rotation=90)
    st.pyplot(fig2)

    # Correlación
    st.write("### Matriz de Correlación")
    corr = df_real[['Nota final', 'Asistencia']].corr()
    st.dataframe(corr)

    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Clustering corregido
    st.write("### Clustering de Estudiantes")
    X = df_real[['Nota final', 'Asistencia']].dropna()
    kmeans = KMeans(n_clusters=3)
    df_real.loc[X.index, 'Cluster'] = kmeans.fit_predict(X)
    st.dataframe(df_real[['Estudiante', 'Nota final', 'Asistencia', 'Cluster']].head())


    # NLP Simulado
    st.write("### Análisis de Sentimiento (Simulado)")
    comentario = "El estudiante mostró avance significativo."
    blob = TextBlob(comentario)
    st.write(f"Comentario: {comentario}")
    st.write(f"Sentimiento: {blob.sentiment}")
