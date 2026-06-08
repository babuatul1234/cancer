import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("Cancer_pred.pkl")

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁"
)

st.title("🫁 Lung Cancer Prediction System")
st.write("Enter patient details and click Predict.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

smoking = st.selectbox("Smoking", [1, 2])
yellow_fingers = st.selectbox("Yellow Fingers", [1, 2])
anxiety = st.selectbox("Anxiety", [1, 2])
peer_pressure = st.selectbox("Peer Pressure", [1, 2])
chronic_disease = st.selectbox("Chronic Disease", [1, 2])
fatigue = st.selectbox("Fatigue", [1, 2])
allergy = st.selectbox("Allergy", [1, 2])
wheezing = st.selectbox("Wheezing", [1, 2])
alcohol_consuming = st.selectbox("Alcohol Consuming", [1, 2])
coughing = st.selectbox("Coughing", [1, 2])
shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2])
chest_pain = st.selectbox("Chest Pain", [1, 2])

# Gender Encoding
gender = 1 if gender == "Male" else 0

if st.button("Predict"):

    input_data = pd.DataFrame([[

        gender,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol_consuming,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain

    ]], columns=[

        'GENDER',
        'AGE',
        'SMOKING',
        'YELLOW_FINGERS',
        'ANXIETY',
        'PEER_PRESSURE',
        'CHRONIC DISEASE',
        'FATIGUE',
        'ALLERGY',
        'WHEEZING',
        'ALCOHOL CONSUMING',
        'COUGHING',
        'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY',
        'CHEST PAIN'

    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Lung Cancer Detected")
    else:
        st.success("✅ No Lung Cancer Detected")
