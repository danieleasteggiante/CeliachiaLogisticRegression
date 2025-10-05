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
    familiaritaceliachia = st.selectbox("Familiarità Celiachia", [0, 1])
    dq2 = st.selectbox("DQ2", [0, 1, 2])
    dq5 = st.selectbox("DQ5", [0, 1, 2])
    dq8 = st.selectbox("DQ8", [0, 1])

def is_all_zeros(record):
    columns_to_be_zero = ['TD1',
    'Pato Tiroide',
    'Deficit \nAccrescimento',
    'Sintomi GI',
    'Familiarità\nCeliachia',
    'DQ2',
    'DQX.5',
    'DQ8',]
    return all(record[col].iloc[0] == 0 for col in columns_to_be_zero)

def genetica_is_absent(record):
    return record['DQ2'].iloc[0] == 0 and record['DQX.5'].iloc[0] == 0 and record['DQ8'].iloc[0] == 0

def adjust_prediction(rec, familiarita, prediction):
    print("adjust_prediction\n", rec, familiarita, prediction)
    if is_all_zeros(rec):
        prediction = 0.1
    if is_all_zeros(rec):
        prediction =  0.01
    if genetica_is_absent(rec):
        prediction = prediction if prediction < 0.33 else 0.33
    if familiarita == 1:
        prediction += 0.03
    return prediction


if st.button("Predici"):
    record = pd.DataFrame([{
        'TD1': td1,
        'Pato Tiroide': patotiroide,
        'Deficit \nAccrescimento': deficitaccrescimento,
        'Sintomi GI': sintomigi,
        'Familiarità\nCeliachia': 0,
        'DQ2': dq2,
        'DQX.5': dq5,
        'DQ8': dq8,
        'Genere_F': 1 if genere == "F" else 0,
        'Genere_M': 1 if genere == "M" else 0
    }])
    pred = model.predict_proba(record)
    prediction_adjusted = adjust_prediction(record, familiaritaceliachia, pred[0][1])
    label = "Basso rischio" if prediction_adjusted < 0.4 else "Alto rischio"
    st.header(f"Predizione: {label} di Celiachia")
    with col3:
        fig = px.bar(x=["Celiachia", "Non Celiachia"], y=[prediction_adjusted, 1-prediction_adjusted],
                     labels={'x': 'Classe', 'y': 'Probabilità'})
        st.plotly_chart(fig)
