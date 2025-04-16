# -*- coding: utf-8 -*-
"""
Enhanced Disease Prediction System
"""

import pickle
import numpy as np
import scipy.sparse.linalg
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Health Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-bottom: 0.1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
    .result-text {
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #050000;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .input-label {
        font-weight: bold;
        color: #ffffff;
    }
    .sidebar .css-1d391kg {
        background-color: #E3F2FD;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #757575;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    diabetes_model = pickle.load(open("Diabetes_model.sav", 'rb'))
    heart_disease_model = pickle.load(open("Heart_model.sav", 'rb'))
    parkinsons_model = pickle.load(open("parkinson_model.sav", 'rb'))
    return diabetes_model, heart_disease_model, parkinsons_model

diabetes_model, Heart_dieases_model, parkinsons_model = load_models()

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

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
    st.markdown("<h2 style='text-align: center;'>Health Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>AI-powered disease prediction</div>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Prediction'],
        icons=['activity', 'heart', 'person'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#050000"},
            "icon": {"color": "#0D47A1", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#d0e1fd"},
            "nav-link-selected": {"background-color": "#1E88E5", "color": "white"},
        }
    )
    
    st.markdown("""
    <div class="info-box">
        <h4 style='text-align: center;'>How it works</h4>
        <p>Our ML models analyze your health data to identify potential health risks early.</p>
        <p style='font-weight: bold; text-align: center;'>Early detection saves lives!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='disclaimer'>This tool is for educational purposes only and not a substitute for professional medical advice.</div>", unsafe_allow_html=True)

# Main content area
if selected == 'Diabetes Prediction':
    st.markdown("<h1 class='main-header'>Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Enter your health parameters below to assess your diabetes risk. Our AI model will analyze these metrics to determine your risk level.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<p class='input-label'>Number of pregnancies</p>", unsafe_allow_html=True)
        Pregnancies = st.text_input('', '0', key='preg')
    
    with col2:
        st.markdown("<p class='input-label'>Glucose level</p>", unsafe_allow_html=True)
        Glucose = st.text_input('', '0', key='glu')
    
    with col3:
        st.markdown("<p class='input-label'>Blood pressure value</p>", unsafe_allow_html=True)
        BloodPressure = st.text_input('', '0', key='bp')
    
    with col1:
        st.markdown("<p class='input-label'>Skin Thickness value</p>", unsafe_allow_html=True)
        SkinThickness = st.text_input('', '0', key='skin')
    
    with col2:
        st.markdown("<p class='input-label'>Insulin level</p>", unsafe_allow_html=True)
        Insulin = st.text_input('', '0', key='insulin')
    
    with col3:
        st.markdown("<p class='input-label'>BMI value</p>", unsafe_allow_html=True)
        BMI = st.text_input('', '0', key='bmi')
    
    with col1:
        st.markdown("<p class='input-label'>Diabetes Pedigree Function value</p>", unsafe_allow_html=True)
        DiabetesPedigreeFunction = st.text_input('', '0', key='dpf')
        
    with col2:
        st.markdown("<p class='input-label'>Age of the Person</p>", unsafe_allow_html=True)
        Age = st.text_input('', '0', key='age')
    
    
    # prediction
    dia_diagnosis = ''
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('Analyze Diabetes Risk'):
        try:
            with st.spinner('Analyzing your data...'):
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
                    dia_diagnosis = 'Low Risk: The analysis indicates you likely do not have diabetes.'
                    result_color = "#4CAF50"  # Green for positive result
                else:
                    dia_diagnosis = 'High Risk: The analysis suggests you may have diabetes. Please consult a healthcare professional.'
                    result_color = "#FF5722"  # Orange for warning
                    
        except ValueError:
            dia_diagnosis = 'Please enter valid numeric values for all fields'
            result_color = "#F44336"  # Red for error
            
        st.markdown(f"""
        <div class="result-text" style="background-color: {result_color}; color: white;">
            {dia_diagnosis}
        </div>
        """, unsafe_allow_html=True)
        
        if diab_prediction[0] == 1:
            st.markdown("""
            <div style="margin-top: 20px;">
                <h4>Next Steps:</h4>
                <ul>
                    <li>Schedule an appointment with your doctor</li>
                    <li>Consider a fasting blood glucose test</li>
                    <li>Review your diet and exercise habits</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    
elif selected == 'Heart Disease Prediction':
    st.markdown("<h1 class='main-header'>Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Enter your cardiac health parameters below. Our AI model will analyze these metrics to evaluate your heart disease risk.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<p class='input-label'>Age</p>", unsafe_allow_html=True)
        age = st.number_input('', min_value=0, max_value=120, key='h_age')
        
    with col2:
        st.markdown("<p class='input-label'>(1=Male, 0=Female)</p>", unsafe_allow_html=True)
        sex = st.number_input('', min_value=0, max_value=1, key='h_sex')
        
    with col3:
        st.markdown("<p class='input-label'>Chest Pain types</p>", unsafe_allow_html=True)
        cp = st.number_input('', min_value=0, max_value=3, key='h_cp')
        
    with col1:
        st.markdown("<p class='input-label'>Resting Blood Pressure</p>", unsafe_allow_html=True)
        trestbps = st.number_input('', min_value=0, key='h_bp')
        
    with col2:
        st.markdown("<p class='input-label'>Serum Cholestoral in mg/dl</p>", unsafe_allow_html=True)
        chol = st.number_input('', min_value=0, key='h_chol')
        
    with col3:
        st.markdown("<p class='input-label'>Fasting Blood Sugar > 120 mg/dl</p>", unsafe_allow_html=True)
        fbs = st.number_input('', min_value=0, max_value=1, key='h_fbs')
        
    with col1:
        st.markdown("<p class='input-label'>Resting Electrocardiographic results</p>", unsafe_allow_html=True)
        restecg = st.number_input('', min_value=0, max_value=2, key='h_restecg')
        
    with col2:
        st.markdown("<p class='input-label'>Maximum Heart Rate achieved</p>", unsafe_allow_html=True)
        thalach = st.number_input('', min_value=0, key='h_thalach')
        
    with col3:
        st.markdown("<p class='input-label'>Exercise Induced Angina</p>", unsafe_allow_html=True)
        exang = st.number_input('', min_value=0, max_value=1, key='h_exang')
        
    with col1:
        st.markdown("<p class='input-label'>ST depression induced by exercise</p>", unsafe_allow_html=True)
        oldpeak = st.number_input('', key='h_oldpeak')
        
    with col2:
        st.markdown("<p class='input-label'>Slope of the peak exercise ST segment</p>", unsafe_allow_html=True)
        slope = st.number_input('', min_value=0, max_value=2, key='h_slope')
        
    with col3:
        st.markdown("<p class='input-label'>Major vessels colored by flourosopy</p>", unsafe_allow_html=True)
        ca = st.number_input('', min_value=0, max_value=4, key='h_ca')
        
    with col1:
        st.markdown("<p class='input-label'>thal: 0 = normal; 1 = fixed defect; 2 = reversable defect</p>", unsafe_allow_html=True)
        thal = st.number_input('', min_value=0, max_value=3, key='h_thal')
        
    heart_diagnosis = ''
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('Analyze Heart Disease Risk'):
        with st.spinner('Analyzing your cardiac data...'):
            heart_prediction = Heart_dieases_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
            
            if (heart_prediction[0] == 1):
                heart_diagnosis = 'High Risk: The analysis suggests you may have heart disease. Please consult a cardiologist.'
                result_color = "#FF5722"  # Orange for warning
            else:
                heart_diagnosis = 'Low Risk: The analysis indicates your heart health appears normal.'
                result_color = "#4CAF50"  # Green for positive result
            
        st.markdown(f"""
        <div class="result-text" style="background-color: {result_color}; color: white;">
            {heart_diagnosis}
        </div>
        """, unsafe_allow_html=True)
        
        if heart_prediction[0] == 1:
            st.markdown("""
            <div style="margin-top: 20px;">
                <h4>Next Steps:</h4>
                <ul>
                    <li>Consult with a cardiologist promptly</li>
                    <li>Consider cardiac stress testing</li>
                    <li>Review your lifestyle factors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    
# Parkinson's Prediction Page
elif selected == "Parkinson Prediction":
    
    st.markdown("<h1 class='main-header'>Parkinson's Disease Risk Assessment</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Enter voice and speech parameters below. These biomarkers can help identify early signs of Parkinson's disease.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        st.markdown("<p class='input-label'>MDVP:Fo(Hz)</p>", unsafe_allow_html=True)
        fo = st.text_input('', '0', key='p_fo')
        
    with col2:
        st.markdown("<p class='input-label'>MDVP:Fhi(Hz)</p>", unsafe_allow_html=True)
        fhi = st.text_input('', '0', key='p_fhi')
        
    with col3:
        st.markdown("<p class='input-label'>MDVP:Flo(Hz)</p>", unsafe_allow_html=True)
        flo = st.text_input('', '0', key='p_flo')
        
    with col4:
        st.markdown("<p class='input-label'>MDVP:Jitter(%)</p>", unsafe_allow_html=True)
        Jitter_percent = st.text_input('', '0', key='p_jit_pct')
        
    with col5:
        st.markdown("<p class='input-label'>MDVP:Jitter(Abs)</p>", unsafe_allow_html=True)
        Jitter_Abs = st.text_input('', '0', key='p_jit_abs')
        
    with col1:
        st.markdown("<p class='input-label'>MDVP:RAP</p>", unsafe_allow_html=True)
        RAP = st.text_input('', '0', key='p_rap')
        
    with col2:
        st.markdown("<p class='input-label'>MDVP:PPQ</p>", unsafe_allow_html=True)
        PPQ = st.text_input('', '0', key='p_ppq')
        
    with col3:
        st.markdown("<p class='input-label'>Jitter:DDP</p>", unsafe_allow_html=True)
        DDP = st.text_input('', '0', key='p_ddp')
        
    with col4:
        st.markdown("<p class='input-label'>MDVP:Shimmer</p>", unsafe_allow_html=True)
        Shimmer = st.text_input('', '0', key='p_shimmer')
        
    with col5:
        st.markdown("<p class='input-label'>MDVP:Shimmer(dB)</p>", unsafe_allow_html=True)
        Shimmer_dB = st.text_input('', '0', key='p_shimmer_db')
        
    with col1:
        st.markdown("<p class='input-label'>Shimmer:APQ3</p>", unsafe_allow_html=True)
        APQ3 = st.text_input('', '0', key='p_apq3')
        
    with col2:
        st.markdown("<p class='input-label'>Shimmer:APQ5</p>", unsafe_allow_html=True)
        APQ5 = st.text_input('', '0', key='p_apq5')
        
    with col3:
        st.markdown("<p class='input-label'>MDVP:APQ</p>", unsafe_allow_html=True)
        APQ = st.text_input('', '0', key='p_apq')
        
    with col4:
        st.markdown("<p class='input-label'>Shimmer:DDA</p>", unsafe_allow_html=True)
        DDA = st.text_input('', '0', key='p_dda')
        
    with col5:
        st.markdown("<p class='input-label'>NHR</p>", unsafe_allow_html=True)
        NHR = st.text_input('', '0', key='p_nhr')
        
    with col1:
        st.markdown("<p class='input-label'>HNR</p>", unsafe_allow_html=True)
        HNR = st.text_input('', '0', key='p_hnr')
        
    with col2:
        st.markdown("<p class='input-label'>RPDE</p>", unsafe_allow_html=True)
        RPDE = st.text_input('', '0', key='p_rpde')
        
    with col3:
        st.markdown("<p class='input-label'>DFA</p>", unsafe_allow_html=True)
        DFA = st.text_input('', '0', key='p_dfa')
        
    with col4:
        st.markdown("<p class='input-label'>spread1</p>", unsafe_allow_html=True)
        spread1 = st.text_input('', '0', key='p_spread1')
        
    with col5:
        st.markdown("<p class='input-label'>spread2</p>", unsafe_allow_html=True)
        spread2 = st.text_input('', '0', key='p_spread2')
        
    with col1:
        st.markdown("<p class='input-label'>D2</p>", unsafe_allow_html=True)
        D2 = st.text_input('', '0', key='p_d2')
        
    with col2:
        st.markdown("<p class='input-label'>PPE</p>", unsafe_allow_html=True)
        PPE = st.text_input('', '0', key='p_ppe')
        
    
    parkinsons_diagnosis = ''
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyze Parkinson's Risk"):
        try:
            with st.spinner('Analyzing your voice parameters...'):
                # Convert to float
                input_params = [float(fo), float(fhi), float(flo), float(Jitter_percent), 
                                float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                                float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                                float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                                float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
                
                parkinsons_prediction = parkinsons_model.predict([input_params])                          
                
                if (parkinsons_prediction[0] == 1):
                    parkinsons_diagnosis = "High Risk: The analysis suggests you may have Parkinson's disease. Please consult a neurologist."
                    result_color = "#FF5722"  # Orange for warning
                else:
                    parkinsons_diagnosis = "Low Risk: The analysis indicates no signs of Parkinson's disease."
                    result_color = "#4CAF50"  # Green for positive result
        except ValueError:
            parkinsons_diagnosis = "Please enter valid numeric values for all fields"
            result_color = "#F44336"  # Red for error
        
        st.markdown(f"""
        <div class="result-text" style="background-color: {result_color}; color: white;">
            {parkinsons_diagnosis}
        </div>
        """, unsafe_allow_html=True)
        
        if 'High Risk' in parkinsons_diagnosis:
            st.markdown("""
            <div style="margin-top: 20px;">
                <h4>Next Steps:</h4>
                <ul>
                    <li>Consult with a neurologist</li>
                    <li>Consider additional diagnostic tests</li>
                    <li>Early intervention can significantly improve outcomes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #f0f2f6;">
    <p style="color: #757575; font-size: 0.9rem;">
        Health Prediction System • © 2025
    </p>
    <p style="color: #757575; font-size: 0.8rem;">
        This tool is for educational purposes only. Always consult with healthcare professionals for medical advice.
    </p>
</div>
""", unsafe_allow_html=True)