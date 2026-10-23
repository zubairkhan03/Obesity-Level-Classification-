import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Obesity Level Predictor", page_icon="⚖️", layout="centered")

MODEL_PATH = Path(__file__).parent / "obesity_svm_model.pkl"

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model(str(MODEL_PATH))
except FileNotFoundError:
    st.error(f"Model file not found at: {MODEL_PATH}\nPut `obesity_svm_model.pkl` next to `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Exact column order the pipeline was trained on
FEATURE_ORDER = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS",
]

st.title("⚖️ Obesity Level Predictor")
st.caption("Enter the person's details and click **Predict**.")

with st.form("input_form"):
    st.subheader("Personal")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        age = st.number_input("Age (years)", min_value=10.0, max_value=100.0, value=25.0, step=0.5)
    with col3:
        family_history = st.selectbox("Family history of overweight", ["yes", "no"])

    st.subheader("Body measurements")
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Height (m)", min_value=1.20, max_value=2.30, value=1.70, step=0.01, format="%.2f")
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)

    st.subheader("Eating habits")
    col1, col2 = st.columns(2)
    with col1:
        favc = st.selectbox("High-calorie food (FAVC)", ["yes", "no"])
        fcvc = st.slider("Vegetable consumption (FCVC, 1-3)", 1.0, 3.0, 2.0, 0.01)
        ncp  = st.slider("Main meals per day (NCP, 1-4)", 1.0, 4.0, 3.0, 0.01)
    with col2:
        caec = st.selectbox("Food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])

    st.subheader("Lifestyle")
    col1, col2, col3 = st.columns(3)
    with col1:
        smoke = st.selectbox("Smoke (SMOKE)", ["no", "yes"])
        scc = st.selectbox("Calorie monitoring (SCC)", ["no", "yes"])
    with col2:
        ch2o = st.slider("Water per day (CH2O, 1-3)", 1.0, 3.0, 2.0, 0.01)
        faf = st.slider("Physical activity (FAF, 0-3)", 0.0, 3.0, 1.0, 0.01)
    with col3:
        tue = st.slider("Tech use (TUE, 0-2)", 0.0, 2.0, 1.0, 0.01)
        mtrans = st.selectbox(
            "Transport (MTRANS)",
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    row = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
    }
    X = pd.DataFrame([row])[FEATURE_ORDER]

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success(f"Predicted class: **{pred}**")

    bmi = weight / (height ** 2)
    st.metric("BMI (informational)", f"{bmi:.1f}")

    st.divider()
    st.caption("Input sent to the model")
    st.dataframe(X, use_container_width=True)