import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
import sklearn
import pickle
import numpy as np

colGPA = pd.read_csv("Promedio.csv")

st.title("Indicadores de promedio")

tab1, tab2 = st.tabs(["Tab1", "Tab2"])

with tab1:
    st.header("Promedio")

    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    ax[0].hist(colGPA["Promedio"])
    conteo = colGPA["Pregrado"].value_counts()
    ax[1].bar(conteo.index, conteo.values)
    ax[2].hist(colGPA["Maestría"])
    conteo = colGPA["Sexo"].value_counts()
    ax[3].bar(conteo.index, conteo.values)
    fig.tight_layout()
    
    st.pyplot(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    sns.scatterplot(data=colGPA, x="Pregrado", y="Promedio", ax=ax[0])
    sns.boxplot(data=colGPA, x="Maestría", y="Promedio", ax=ax[1])
    sns.scatterplot(data=colGPA, x="Sexo", y="Promedio", ax=ax[2])
    fig.tight_layout()
    
    st.pyplot(fig)

with open("model.pickle", "rb") as f:
  modelo = pickle.load(f)
with tab2:
    junior = st.selectbox("Pregrado", ["Sí", "No"])
    if junior == "Sí":
       junior = 1
    else:
      junior = 0
    senior = st.selectbox("Maestría", ["Sí", "No"])
    if senior == "Sí":
      senior = 1
    else:
      senior = 0
    male = st.selectbox("Sexo", ["Masc", "Fem"])
    if male == "Masc":
      male = 1
    else:
      male = 0
    
    if st.button("Predecir"):
      pred = modelo.predict(np.array([[junior, senior, male]]))
      st.write(f"Su promedio sería {round(pred[0], 1)}")


