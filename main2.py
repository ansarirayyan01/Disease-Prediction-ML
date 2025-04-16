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

# Set page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    body, .main, .stApp {
        background: linear-gradient(135deg, #ff1744 0%, #212121 100%) !important;
    }
    .main {
        max-width: 1100px;
        margin: 0 auto;
        padding: 2rem 0.5rem;
    }
    .sidebar-content, .css-1d391kg, .css-1wrcr25 {
        background: #232f34 !important;
        color: #fff !important;
        border-radius: 0 12px 12px 0;
        box-shadow: 2px 0 8px rgba(44,62,80,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 7px;
        height: 2.5em;
        font-size: 1.08em;
        background: linear-gradient(90deg, #4fc3f7 0%, #1976d2 100%);
        color: #fff;
        border: none;
        box-shadow: 0 2px 6px rgba(25,118,210,0.12);
        font-weight: 600;
        transition: background 0.2s, box-shadow 0.2s, transform 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1976d2 0%, #4fc3f7 100%);
        box-shadow: 0 4px 12px rgba(25,118,210,0.18);
        transform: translateY(-1px) scale(1.01);
    }
    .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #f9fbfd;
        border-radius: 6px;
        border: 1px solid #b0bec5;
        padding: 8px;
        font-size: 1em;
        color: #232f34;
        box-shadow: 0 1px 3px rgba(33,150,243,0.03);
    }
    .stNumberInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 2px rgba(33,150,243,0.13);
    }
    .card {
        background: #fff;
        border-radius: 15px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: 0 2px 12px rgba(33,150,243,0.07);
        border: none;
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .card:hover {
        box-shadow: 0 6px 24px rgba(33,150,243,0.12);
        transform: translateY(-2px) scale(1.01);
    }
    h1, h2, h3, h4 {
        color: #1976d2;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-weight: 600;
    }
    h1 {
        text-align: center;
        margin-bottom: 18px;
        font-size: 2.1em;
    }
    h2 {
        margin-top: 12px;
        margin-bottom: 10px;
        font-size: 1.35em;
    }
    h3 {
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 1.1em;
        color: #232f34;
    }
    label, .stNumberInput label, .stSelectbox label {
        color: #ffffff;
        font-size: 1em;
        font-weight: 500;
        opacity: 0.85;
    }
    ::-webkit-scrollbar {
        width: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #1976d2;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1565c0;
    }
    .stAlert {
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
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

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: white; margin-bottom: 30px;'>Disease Prediction System</h1>
        <p style='color: white;'>Select a disease to predict:</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        None,
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinson Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0,
        menu_icon="hospital",
        styles={
            "container": {"padding": "5px", "background-color": "rgba(255,255,255,0.1)"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.2)"},
        }
    )

# Main content
if (selected == 'Diabetes Prediction'):
    st.markdown("""
    <div class='card' style='background: #e8eaf6; border-left: 4px solid #1a237e;'>
        <h1 style='color: #1a237e; text-align: center;'>Diabetes Prediction 🩸</h1>
        <p style='text-align: center; color: #bf0000; font-size: 1.1em;'>
            Predict the likelihood of diabetes based on health parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='margin-top: 15px; background: #ffffff; border-left: 4px solid #1a237e;'>
        <h2 style='color: #1a237e;'>Patient Information</h2>
        <p style='color: #bf0000;'>Please enter the patient's health parameters below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Basic Information</h3>
        """, unsafe_allow_html=True)
        Pregnancies = st.number_input('Number of pregnancies', min_value=0, max_value=20, value=0, help="Number of times pregnant")
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=0, help="Age in years")
        BMI = st.number_input('BMI value', min_value=0.0, max_value=100.0, value=0.0, help="Body Mass Index")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Blood Tests</h3>
        """, unsafe_allow_html=True)
        Glucose = st.number_input('Glucose level', min_value=0, max_value=200, value=0, help="Plasma glucose concentration")
        BloodPressure = st.number_input('Blood pressure value', min_value=0, max_value=200, value=0, help="Diastolic blood pressure (mm Hg)")
        Insulin = st.number_input('Insulin level', min_value=0, max_value=1000, value=0, help="2-Hour serum insulin (mu U/ml)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Additional Measurements</h3>
        """, unsafe_allow_html=True)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=0, help="Triceps skin fold thickness (mm)")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=3.0, value=0.0, help="Diabetes pedigree function")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button('Get Diabetes Prediction', type="primary"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        input_data_scaled = scaler.transform(input_data)
        diab_prediction = diabetes_model.predict(input_data_scaled)
        
        if (diab_prediction[0] == 0):
            st.markdown("""
            <div class='card' style='background: #e8f5e9; border-left: 4px solid #2e7d32; color: #1b5e20; text-align: center; padding: 15px;'>
                <h2 style='color: #1b5e20;'>✅ Not Diabetic</h2>
                <p style='font-size: 1.1em;'>The patient shows no signs of diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='background: #ffebee; border-left: 4px solid #c62828; color: #b71c1c; text-align: center; padding: 15px;'>
                <h2 style='color: #b71c1c;'>⚠️ Diabetic</h2>
                <p style='font-size: 1.1em;'>The patient shows signs of diabetes. Please consult a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)

if (selected == 'Heart Disease Prediction'):
    st.markdown("""
    <div class='card' style='background: #e8eaf6; border-left: 4px solid #1a237e;'>
        <h1 style='color: #1a237e; text-align: center;'>Heart Disease Prediction ❤️</h1>
        <p style='text-align: center; color: #424242; font-size: 1.1em;'>
            Predict the likelihood of heart disease based on health parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='margin-top: 15px; background: #ffffff; border-left: 4px solid #1a237e;'>
        <h2 style='color: #1a237e;'>Patient Information</h2>
        <p style='color: #424242;'>Please enter the patient's health parameters below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Personal Information</h3>
        """, unsafe_allow_html=True)
        age = st.number_input('Age', min_value=0, max_value=120, value=0)
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 0
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Medical History</h3>
        """, unsafe_allow_html=True)
        cp = st.selectbox('Chest Pain Type', 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                         help="Type of chest pain experienced")
        cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=0)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, value=0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Test Results</h3>
        """, unsafe_allow_html=True)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0
        restecg = st.selectbox('Resting ECG Results', 
                              ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        restecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
        thalach = st.number_input('Maximum Heart Rate', min_value=0, max_value=300, value=0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Exercise Test</h3>
        """, unsafe_allow_html=True)
        exang = st.selectbox('Exercise Induced Angina', ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0
        oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=0.0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Additional Parameters</h3>
        """, unsafe_allow_html=True)
        slope = st.selectbox('Slope of Peak Exercise', 
                           ["Upsloping", "Flat", "Downsloping"])
        slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
        ca = st.number_input('Number of Major Vessels', min_value=0, max_value=4, value=0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Thalassemia</h3>
        """, unsafe_allow_html=True)
        thal = st.selectbox('Thalassemia', 
                          ["Normal", "Fixed Defect", "Reversible Defect"])
        thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button('Get Heart Disease Prediction', type="primary"):
        heart_prediction = Heart_dieases_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        if (heart_prediction[0] == 1):
            st.markdown("""
            <div class='card' style='background: #e8f5e9; border-left: 4px solid #2e7d32; color: #1b5e20; text-align: center; padding: 15px;'>
                <h2 style='color: #1b5e20;'>⚠️ At Risk</h2>
                <p style='font-size: 1.1em;'>The patient is at risk of heart disease. Please consult a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='background: #ffebee; border-left: 4px solid #c62828; color: #b71c1c; text-align: center; padding: 15px;'>
                <h2 style='color: #b71c1c;'>✅ Not At Risk</h2>
                <p style='font-size: 1.1em;'>The patient shows no signs of heart disease.</p>
            </div>
            """, unsafe_allow_html=True)

if (selected == "Parkinson Prediction"):
    st.markdown("""
    <div class='card' style='background: #e8eaf6; border-left: 4px solid #1a237e;'>
        <h1 style='color: #1a237e; text-align: center;'>Parkinson's Disease Prediction 🧠</h1>
        <p style='text-align: center; color: #424242; font-size: 1.1em;'>
            Predict the likelihood of Parkinson's disease based on voice measurements
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='margin-top: 15px; background: #ffffff; border-left: 4px solid #1a237e;'>
        <h2 style='color: #1a237e;'>Voice Parameters</h2>
        <p style='color: #424242;'>Please enter the patient's voice measurements below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Fundamental Frequency</h3>
        """, unsafe_allow_html=True)
        fo = st.number_input('MDVP:Fo(Hz)', format="%.2f")
        fhi = st.number_input('MDVP:Fhi(Hz)', format="%.2f")
        flo = st.number_input('MDVP:Flo(Hz)', format="%.2f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Jitter Measurements</h3>
        """, unsafe_allow_html=True)
        Jitter_percent = st.number_input('MDVP:Jitter(%)', format="%.4f")
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', format="%.4f")
        RAP = st.number_input('MDVP:RAP', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Additional Jitter</h3>
        """, unsafe_allow_html=True)
        PPQ = st.number_input('MDVP:PPQ', format="%.4f")
        DDP = st.number_input('Jitter:DDP', format="%.4f")
        Shimmer = st.number_input('MDVP:Shimmer', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Shimmer Measurements</h3>
        """, unsafe_allow_html=True)
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', format="%.4f")
        APQ3 = st.number_input('Shimmer:APQ3', format="%.4f")
        APQ5 = st.number_input('Shimmer:APQ5', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Additional Parameters</h3>
        """, unsafe_allow_html=True)
        APQ = st.number_input('MDVP:APQ', format="%.4f")
        DDA = st.number_input('Shimmer:DDA', format="%.4f")
        NHR = st.number_input('NHR', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Voice Analysis</h3>
        """, unsafe_allow_html=True)
        HNR = st.number_input('HNR', format="%.4f")
        RPDE = st.number_input('RPDE', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col7:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Additional Metrics</h3>
        """, unsafe_allow_html=True)
        DFA = st.number_input('DFA', format="%.4f")
        spread1 = st.number_input('spread1', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col8:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Spread Parameters</h3>
        """, unsafe_allow_html=True)
        spread2 = st.number_input('spread2', format="%.4f")
        D2 = st.number_input('D2', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col9:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>Final Metrics</h3>
        """, unsafe_allow_html=True)
        PPE = st.number_input('PPE', format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Get Parkinson's Prediction", type="primary"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        
        if (parkinsons_prediction[0] == 1):
            st.markdown("""
            <div class='card' style='background: #e8f5e9; border-left: 4px solid #2e7d32; color: #1b5e20; text-align: center; padding: 15px;'>
                <h2 style='color: #1b5e20;'>⚠️ Parkinson's Detected</h2>
                <p style='font-size: 1.1em;'>The patient shows signs of Parkinson's disease. Please consult a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='background: #ffebee; border-left: 4px solid #c62828; color: #b71c1c; text-align: center; padding: 15px;'>
                <h2 style='color: #b71c1c;'>✅ No Parkinson's</h2>
                <p style='font-size: 1.1em;'>The patient shows no signs of Parkinson's disease.</p>
            </div>
            """, unsafe_allow_html=True)
    
    

