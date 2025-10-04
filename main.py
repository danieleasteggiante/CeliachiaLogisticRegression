import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load('logistic_model.pkl')

st.title("Predictive Model for Celiac Disease")
st.write("""
Questa applicazione utilizza un modello di regressione logistica per predire la probabilità
di Celiachia basandosi su vari fattori clinici e genetici.
""")

col1, col2, col3 = st.columns(3)

with col1:
    td1 = st.selectbox("TD1", [0, 1])
    patotiroide = st.selectbox("Pato Tiroide", [0,1])
    deficitaccrescimento = st.selectbox("Deficit Accrescimento", [0, 1])
    sintomigi = st.selectbox("Sintomi GI", [0, 1])
    genere = st.selectbox("Genere", ["F", "M"])

with col2:
    familiaritaceliachia = st.selectbox("Familiarità Celiachia", [0, 1,2])
    familiaritaceliachia = 2 if familiaritaceliachia > 0 else 0
    dq2 = st.selectbox("DQ2", [0, 1, 2])
    dq5 = st.selectbox("DQ5", [0, 1, 2])
    dq8 = st.selectbox("DQ8", [0, 1])

if st.button("Predici"):
    record = pd.DataFrame([{
        'TD1': td1,
        'Pato Tiroide': patotiroide,
        'Deficit \nAccrescimento': deficitaccrescimento,
        'Sintomi GI': sintomigi,
        'Familiarità\nCeliachia': familiaritaceliachia,
        'DQ2': dq2,
        'DQX.5': dq5,
        'DQ8': dq8,
        'Genere_F': 1 if genere == "F" else 0,
        'Genere_M': 1 if genere == "M" else 0
    }])
    pred = model.predict_proba(record)
    st.header(f"Predizione: {pred[0][1]:.2f} probabilità di Celiachia")
    with col3:
        fig = px.bar(x=["Celiachia", "Non Celiachia"], y=[pred[0][1], pred[0][0]],
                     labels={'x': 'Classe', 'y': 'Probabilità'})
        st.plotly_chart(fig)
