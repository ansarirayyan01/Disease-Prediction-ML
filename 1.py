# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
import scipy.sparse.linalg
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler


diabetes_model = pickle.load(open("Diabetes_model.sav",'rb'))
Heart_dieases_model = pickle.load(open("Heart_model.sav",'rb'))
parkinsons_model = pickle.load(open("parkinson_model.sav",'rb'))

# Create a new scaler (we'll fit it with some sample data)
scaler = StandardScaler()
# Sample data to fit the scaler (using typical ranges)
sample_data = np.array([
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # diabetic
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],   # non-diabetic
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],   # diabetic
    [1, 89, 66, 23, 94, 28.1, 0.167, 21],  # non-diabetic
    [0, 137, 40, 35, 168, 43.1, 2.288, 33] # diabetic
])
scaler.fit(sample_data)


with st.sidebar:
    selected = option_menu('Dieases predictive system',
                           ['Diabetes Prediction',
                            'Heart Disease prediction',
                            'Parkinson Prediction'],
                           icons = ['activity','heart','person'],
                           default_index = 0)


if (selected == 'Diabetes Prediction'):
    st.title('Diabetes prediction using ML')
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of pregnancies', '0')
    
    with col2:
        Glucose = st.text_input('Glucose level', '0')
    
    with col3:
        BloodPressure = st.text_input('Blood pressure value', '0')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', '0')
    
    with col2:
        Insulin = st.text_input('Insulin level', '0')
    
    with col3:
        BMI = st.text_input('BMI value', '0')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '0')
        
    with col2:
        Age = st.text_input('Age of the Person', '0')
    
    
    # prediction
    dia_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            # Convert all inputs to float
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            
            # Reshape for scaler
            input_array = np.array(input_data).reshape(1, -1)
            
            # Scale the input data
            scaled_data = scaler.transform(input_array)
            
            # Make prediction
            diab_prediction = diabetes_model.predict(scaled_data)
            
            if (diab_prediction[0] == 0):
                dia_diagnosis = 'The person is not diabetic'
            else:
                dia_diagnosis = 'The person is diabetic'
                
        except ValueError:
            dia_diagnosis = 'Please enter valid numeric values for all fields'
            
    st.success(dia_diagnosis)
    
    
if (selected == 'Heart Disease prediction'):
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex=st.number_input("(1=Male,0=Female)")
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
    heart_diagnosis = ''
    
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = Heart_dieases_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
    
    
# Parkinson's Prediction Page
if (selected == "Parkinson Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    parkinsons_diagnosis = ''
    
  
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)