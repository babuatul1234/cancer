import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("Cancer_pred.pkl")

# Load the scaler (you'll need to save this during training)
# For now, we'll create a scaler with the same parameters as in training
def get_scaler():
    scaler = StandardScaler()
    # The scaler was fitted on AGE column with these parameters from training
    scaler.mean_ = np.array([55.171448])
    scaler.scale_ = np.array([14.728302])
    return scaler

# Title and description
st.title("🏥 Lung Cancer Risk Prediction")
st.markdown("""
This application predicts the risk of lung cancer based on various symptoms and lifestyle factors.
Please fill in the information below to get a prediction.
""")

# Create two columns for input layout
col1, col2 = st.columns(2)

# Input fields
with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 30, 80, 55)
    
    st.subheader("Lifestyle Factors")
    smoking = st.selectbox("Smoking", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    alcohol_consuming = st.selectbox("Alcohol Consumption", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    
    st.subheader("Physical Symptoms")
    fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")

with col2:
    st.subheader("Psychological Factors")
    anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    
    st.subheader("Medical History")
    yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
    wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")

# Convert inputs to model format
def prepare_input_data():
    # Encode gender
    gender_encoded = 1 if gender == "Male" else 0
    
    # Create feature array
    features = [
        gender_encoded,          # GENDER
        age,                     # AGE
        smoking,                 # SMOKING
        yellow_fingers,          # YELLOW_FINGERS
        anxiety,                 # ANXIETY
        peer_pressure,           # PEER_PRESSURE
        chronic_disease,         # CHRONIC_DISEASE
        fatigue,                 # FATIGUE
        allergy,                 # ALLERGY
        wheezing,                # WHEEZING
        alcohol_consuming,       # ALCOHOL_CONSUMING
        coughing,                # COUGHING
        shortness_of_breath,     # SHORTNESS_OF_BREATH
        swallowing_difficulty,   # SWALLOWING_DIFFICULTY
        chest_pain               # CHEST_PAIN
    ]
    
    return np.array(features).reshape(1, -1)

# Prediction button
if st.button("🔍 Predict Lung Cancer Risk", type="primary"):
    with st.spinner("Analyzing your symptoms..."):
        # Load model
        model = load_model()
        
        # Prepare input data
        input_data = prepare_input_data()
        
        # Scale age (as done in training)
        scaler = get_scaler()
        input_data[:, 1:2] = scaler.transform(input_data[:, 1:2])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Create columns for result display
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK")
                st.markdown("""
                **The model suggests a high probability of lung cancer risk.**
                
                **Recommendations:**
                - Consult a healthcare professional immediately
                - Get a comprehensive medical checkup
                - Consider chest X-ray or CT scan
                - Quit smoking if applicable
                - Maintain a healthy lifestyle
                """)
            else:
                st.success("✅ LOW RISK")
                st.markdown("""
                **The model suggests a low probability of lung cancer risk.**
                
                **Recommendations:**
                - Maintain a healthy lifestyle
                - Regular health checkups
                - Avoid smoking and limit alcohol
                - Stay physically active
                - Report any persistent symptoms to your doctor
                """)
        
        with res_col2:
            # Display probability gauge
            prob_percent = prediction_proba[1] * 100
            st.markdown(f"### Risk Probability: {prob_percent:.1f}%")
            
            # Create a simple progress bar for probability
            st.progress(int(prob_percent))
            
            # Display feature importance info
            st.info("""
            **Note:** This prediction is based on a machine learning model 
            trained on historical data. It is for informational purposes only 
            and should not replace professional medical advice.
            """)

# Add sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses a **K-Nearest Neighbors (KNN)** machine learning model 
    to predict lung cancer risk based on various symptoms and lifestyle factors.
    
    ### Model Performance
    - **Accuracy:** ~53.3%
    - The model considers 15 different features including:
      - Demographics (Age, Gender)
      - Lifestyle (Smoking, Alcohol)
      - Physical symptoms
      - Medical history
      - Psychological factors
    
    ### Important Note
    ⚠️ This is a predictive tool and **not a diagnostic device**. 
    Always consult with qualified healthcare professionals for medical advice.
    """)
    
    st.header("📋 Input Guide")
    st.markdown("""
    - **1 = No** / **2 = Yes** for most symptoms
    - **Age** ranges from 30-80 years
    - Provide accurate information for best results
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center>Developed for Healthcare Analytics | Lung Cancer Risk Prediction Model</center>",
    unsafe_allow_html=True
)