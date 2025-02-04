import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


st.set_page_config(page_title='Prediction of Disease Outbreaks', layout='wide', page_icon="Doctor")

# Load models
diabetes_model = pickle.load(open(r"C:\Users\lenovo\Documents\New folder\datasets\diabetes_model.sav", 'rb'))
heart_model = pickle.load(open(r"C:\Users\lenovo\Documents\New folder\datasets\heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open(r"C:\Users\lenovo\Documents\New folder\datasets\parkinsons_model.sav", 'rb'))

# Sidebar Menu
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System', ['Diabetes Prediction', 'Heart Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level')
    with col3:
        Bloodpressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(Bloodpressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = 'Please ensure all inputs are valid numbers.'

    st.success(diab_diagnosis)

# Heart Prediction
elif selected == 'Heart Prediction':
    st.title('Heart Prediction using ML')

    # Layout for input fields
    col1, col2, col3, col4 = st.columns(4)

    # Input fields for user data
    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex (1 for Male, 0 for Female)")  # Input expected to be 0 or 1
    with col3:
        cp = st.text_input("Chest Pain Type (0-3)")
    with col4:
        trestbps = st.text_input("Resting Blood Pressure (mm Hg)")
    with col1:
        chol = st.text_input("Serum Cholesterol (mg/dL)")
    with col2:
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dL")
    with col3:
        restecg = st.text_input("Resting ECG Results (0-2)")
    with col4:
        thalach = st.text_input("Maximum Heart Rate Achieved")
    with col1:
        exang = st.text_input("Exercise Induced Angina")
    with col2:
        oldpeak = st.text_input("ST Depression (Oldpeak)")
    with col3:
        slope = st.text_input("Slope of Peak Exercise ST Segment (0-2)")
    with col4:
        ca = st.text_input("Number of Major Vessels (0-4)")
    with col1:
        thal = st.text_input("Thalassemia (0-3)")

    heart_diagnosis = ''
    if st.button('Heart Test Result'):
        try:
            # Convert inputs to appropriate types
            age = int(age)
            # Handle sex input correctly
            if sex.lower() == 'male':
                sex = 1
            elif sex.lower() == 'female':
                sex = 0
            else:
                sex = int(sex)  # if the user enters 0 or 1 directly
            cp = int(cp)
            trestbps = int(trestbps)
            chol = int(chol)
            fbs = int(fbs)
            restecg = int(restecg)
            thalach = int(thalach)
            exang = int(exang)
            oldpeak = float(oldpeak)
            slope = int(slope)
            ca = int(ca)
            thal = int(thal)

            # Prepare the input as a 2D array for prediction
            user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            # Make the prediction
            heart_prediction = heart_model.predict(user_input)

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person has no heart disease'

        except ValueError:
            heart_diagnosis = 'Please ensure all inputs are valid numbers.'

    # Show the result
    st.success(heart_diagnosis)




elif selected == 'Parkinsons Prediction':
    st.title('Parkinson\'s Disease Prediction using ML')

    # Layout for input fields
    col1, col2, col3, col4 = st.columns(4)

    # Input fields for user data
    with col1:
        Fo = st.text_input('Fo')
    with col2:
        Fhi = st.text_input('Fhi')
    with col3:
        Flo = st.text_input('Flo')
    with col4:
        Jitter_percent = st.text_input('Jitter_percent')
    with col1:
        Jitter_abs = st.text_input('Jitter_abs')
    with col2:
        RAP = st.text_input('RAP')
    with col3:
        PPQ = st.text_input('PPQ')
    with col4:
        DDP = st.text_input('DDP')
    with col1:
        Shimmer = st.text_input('Shimmer')
    with col2:
        Shimmer_db = st.text_input('Shimmer_db')
    with col3:
        APQ3 = st.text_input('APQ3')
    with col4:
        APQ5 = st.text_input('APQ5')
    with col1:
        APQ = st.text_input('APQ')
    with col2:
        DDA = st.text_input('DDA')
    with col3:
        NHR = st.text_input('NHR')
    with col4:
        HNR = st.text_input('HNR')
    with col1:
        RPDE = st.text_input('RPDE')
    with col2:
        DFA = st.text_input('DFA')
    with col3:
        spread1 = st.text_input('spread1')
    with col4:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    if st.button('Parkinson\'s Test Result'):
        try:
            # Convert inputs to appropriate types (floats)
            user_input = [float(Fo), float(Fhi), float(Flo), float(Jitter_percent), float(Jitter_abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_db),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
            
            # Make the prediction
            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'The person has Parkinson\'s disease'
            else:
                parkinsons_diagnosis = 'The person does not have Parkinson\'s disease'

        except ValueError:
            parkinsons_diagnosis = 'Please ensure all inputs are valid numbers.'

    st.success(parkinsons_diagnosis)