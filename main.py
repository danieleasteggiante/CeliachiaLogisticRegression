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

error = ''

map_si_no = {"NO": 0, "SI": 1}
map_assente_presente = {"ASSENTE": 0, "PRESENTE": 1}

with col1:
    td1 = st.selectbox("TD1", ["NO", "SI"])
    patotiroide = st.selectbox("Pato Tiroide", ["NO","SI"])
    deficitaccrescimento = st.selectbox("Deficit Accrescimento", ["NO", "SI"])
    sintomigi = st.selectbox("Sintomi GI", ["NO", "SI"])
    genere = st.selectbox("Genere", ["F", "M"])

with col2:
    familiaritaceliachia = st.selectbox("Familiarità Celiachia", ["NO", "SI"])
    dq2 = st.selectbox("DQ2", ["ASSENTE", 1, 2])
    dq5 = st.selectbox("DQ5", ["ASSENTE", "PRESENTE"])
    dq8 = st.selectbox("DQ8", ["ASSENTE", "PRESENTE"])

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

def genetica_is_not_possible(record):
    return record['DQ2'].iloc[0] == 2 and  record['DQ8'].iloc[0] == 1

def check_errors(record):
    if genetica_is_not_possible(record):
        return "Errore: Combinazione genetica non possibile."
    return ''


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
    return min(prediction, 1.0)


def create_label(prediction, error):
    if error != '':
        return error
    risk =  "Basso rischio" if prediction < 0.4 else "Alto rischio"
    return f"Predizione: {risk} di Celiachia"


if st.button("Predici"):
    record = pd.DataFrame([{
        'TD1': map_si_no[td1],
        'Pato Tiroide': map_si_no[patotiroide],
        'Deficit \nAccrescimento': map_si_no[deficitaccrescimento],
        'Sintomi GI': map_si_no[sintomigi],
        'Familiarità\nCeliachia': 0,
        'DQ2': 0 if dq2 == "ASSENTE" else dq2,
        'DQX.5': map_assente_presente[dq5],
        'DQ8': map_assente_presente[dq8],
        'Genere_F': 1 if genere == "F" else 0,
        'Genere_M': 1 if genere == "M" else 0
    }])
    pred = model.predict_proba(record)
    prediction_adjusted = adjust_prediction(record, familiaritaceliachia, pred[0][1])
    errors = check_errors(record)
    label = create_label(prediction_adjusted, errors)
    st.header(label)
    with col3:
        if errors == '':
            fig = px.bar(x=["Celiachia", "Non Celiachia"], y=[prediction_adjusted, 1-prediction_adjusted],
                         labels={'x': 'Classe', 'y': 'Probabilità'})
            st.plotly_chart(fig)
