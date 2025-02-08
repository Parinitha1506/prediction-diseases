import streamlit as st
import numpy as np
import joblib

# Load trained models (you need to train and save these beforehand)
try:
    diabetes_model = joblib.load("diabetes_model.sav")
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("The file 'diabetes_model.sav' was not found. Please ensure it exists in the specified location.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

try:
    heart_model = joblib.load("heart_model.sav")
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("The file 'heart_model.sav' was not found. Please ensure it exists in the specified location.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

try:
    parkinsons_model = joblib.load("parkinsons_model.sav")
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("The file 'parkinsons_model.sav' was not found. Please ensure it exists in the specified location.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# App title
st.sidebar.title("Prediction of Disease Outbreaks System")
st.title("Disease Prediction using ML")

# Navigation menu
menu = st.sidebar.radio("Select Prediction Type:", ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"])

# Diabetes Prediction
if menu == "Diabetes Prediction":
    st.header("Diabetes Prediction using ML")

    # Input fields for diabetes prediction
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=85)
    blood_pressure = st.number_input("Blood Pressure Value", min_value=0, max_value=122, value=80)
    skin_thickness = st.number_input("Skin Thickness Value", min_value=0, max_value=99, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=85)
    bmi = st.number_input("BMI Value", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function Value", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age of the Person", min_value=0, max_value=120, value=30)

    # Prediction
    if st.button("Predict Diabetes"):
        diabetes_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        diabetes_prediction = diabetes_model.predict(diabetes_input)
        result = "Person has Diabetic Disease" if diabetes_prediction[0] == 1 else "Person has Non-Diabetic Disease"
        st.success(f"Diabetes Prediction: {result}")

# Heart Disease Prediction
elif menu == "Heart Disease Prediction":
    st.header("Heart Disease Prediction using ML")

    # Input fields for heart disease prediction
    age = st.number_input("Age", min_value=0, value=45, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1, step=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120, step=1)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, value=200, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["0", "1"])
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=0, step=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150, step=1)
    exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, value=1.0, step=0.1)
    slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1, step=1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0, step=1)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=2, step=1)

    # Convert categorical input to numerical values
    sex = 1 if sex == "Male" else 0
    fbs = int(fbs)
    exang = 1 if exang == "Yes" else 0

    # Prediction
    if st.button("Predict Heart Disease"):
        heart_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_prediction = heart_model.predict(heart_input)
        result = "Person has Heart Disease" if heart_prediction[0] == 1 else "Person has No Heart Disease"
        st.success(f"Heart Disease Prediction: {result}")


    # Parkinsons Prediction
elif menu == "Parkinson's Prediction":
    st.header("Parkinson's Disease Prediction using ML")

    # Input fields for Parkinson's prediction (excluding name)
    Fo = st.number_input("MDVP:Fo (Hz)", min_value=0.0, value=142.16, step=0.1)
    Fhi = st.number_input("MDVP:Fhi (Hz)", min_value=0.0, value=217.45, step=0.1)
    Flo = st.number_input("MDVP:Flo (Hz)", min_value=0.0, value=83.0, step=0.1)
    Jitter_percent = st.number_input("MDVP:Jitter (%)", min_value=0.0, value=0.01, step=0.001)
    Jitter_abs = st.number_input("MDVP:Jitter (Abs)", min_value=0.0, value=0.0001, step=0.00001)
    RAP = st.number_input("MDVP:RAP", min_value=0.0, value=0.01, step=0.001)
    PPQ = st.number_input("MDVP:PPQ", min_value=0.0, value=0.01, step=0.001)
    DDP = st.number_input("Jitter:DDP", min_value=0.0, value=0.01, step=0.001)
    Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, value=0.015, step=0.001)
    Shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, value=0.12, step=0.01)
    APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, value=0.01, step=0.001)
    APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, value=0.01, step=0.001)
    APQ = st.number_input("MDVP:APQ", min_value=0.0, value=0.01, step=0.001)
    DDA = st.number_input("Shimmer:DDA", min_value=0.0, value=0.02, step=0.001)
    NHR = st.number_input("NHR", min_value=0.0, value=0.01, step=0.001)
    HNR = st.number_input("HNR (dB)", min_value=0.0, value=25.17, step=0.1)
    RPDE = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    DFA = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    spread1 = st.number_input("Spread1", min_value=-10.0, max_value=10.0, value=-5.3, step=0.1)
    spread2 = st.number_input("Spread2", min_value=-10.0, max_value=10.0, value=0.2, step=0.1)
    D2 = st.number_input("D2", min_value=0.0, max_value=10.0, value=2.2, step=0.1)
    PPE = st.number_input("PPE", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    # Create an input array excluding the 'name' field
    parkinsons_input = np.array([[Fo, Fhi, Flo, Jitter_percent, Jitter_abs, RAP, PPQ, DDP, Shimmer, Shimmer_db, 
                                  APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2,PPE]])
    
    try:
        # If the model has preprocessing steps, it will have a 'steps' attribute
        if hasattr(parkinsons_model, 'steps'):
            st.write("Model is likely a pipeline with preprocessing steps.")
            st.write(f"Pipeline steps: {parkinsons_model.steps}")
        else:
            st.write("Model is not a pipeline.")
    except Exception as e:
        st.write(f"Error inspecting the model: {e}")
    
    # Print the shape of the input data and expected features
    st.write(f"Shape of input data: {parkinsons_input.shape}")
    st.write(f"Expected number of features: {parkinsons_model.n_features_in_}")

    # Prediction
    if st.button("Predict Parkinson's"):
        
            parkinsons_prediction = parkinsons_model.predict(parkinsons_input)
            result = "Person Has Parkinson's Disease" if parkinsons_prediction[0] == 1 else "Person has No Parkinson's"
            st.success(f"Parkinson's Prediction: {result}")