
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Cancer_pred.pkl")

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁"
)

st.title("🫁 Lung Cancer Prediction System")
st.write("Enter patient details and click Predict")

# Get feature names from trained model
features = list(model.feature_names_in_)

# User inputs
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

# Convert to model values
smoking = 2 if smoking == "Yes" else 1
yellow_fingers = 2 if yellow_fingers == "Yes" else 1
anxiety = 2 if anxiety == "Yes" else 1
peer_pressure = 2 if peer_pressure == "Yes" else 1
chronic_disease = 2 if chronic_disease == "Yes" else 1
fatigue = 2 if fatigue == "Yes" else 1
allergy = 2 if allergy == "Yes" else 1
wheezing = 2 if wheezing == "Yes" else 1
alcohol_consuming = 2 if alcohol_consuming == "Yes" else 1
coughing = 2 if coughing == "Yes" else 1
shortness_of_breath = 2 if shortness_of_breath == "Yes" else 1
swallowing_difficulty = 2 if swallowing_difficulty == "Yes" else 1
chest_pain = 2 if chest_pain == "Yes" else 1

# IMPORTANT:
# Change this if LabelEncoder encoded differently
gender_encoded = 1 if gender == "Male" else 0

if st.button("Predict"):

    values = {
        "GENDER": gender_encoded,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC DISEASE": chronic_disease,
        "CHRONIC_DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL CONSUMING": alcohol_consuming,
        "ALCOHOL_CONSUMING": alcohol_consuming,
        "COUGHING": coughing,
        "SHORTNESS OF BREATH": shortness_of_breath,
        "SHORTNESS_OF_BREATH": shortness_of_breath,
        "SWALLOWING DIFFICULTY": swallowing_difficulty,
        "SWALLOWING_DIFFICULTY": swallowing_difficulty,
        "CHEST PAIN": chest_pain,
        "CHEST_PAIN": chest_pain
    }

    # Build row dynamically using model feature names
    row = [values[col] for col in features]

    input_data = pd.DataFrame([row], columns=features)

    #st.write("Model expects:")
    #st.write(features)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Lung Cancer Detected")
    else:
        st.success("✅ No Lung Cancer Detected")

