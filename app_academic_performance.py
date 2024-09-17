# -*- coding: utf-8 -*-
"""
Created on Friday September 13 15:29:50 2024

@author: Alphatech
"""

import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
#loading the saved models
#1 - diabetetic model
performance_model = joblib.load('academic_student_performance.pkl')

 


#sidebar for navigation
with st.sidebar:
    selected = option_menu('Student Academic Performance Prediction System',['Academic Performance'],
                           icons = ['person'],default_index=0)
    
#Academic Performance Prediction page
if(selected == 'Academic Performance'):
    html_temp = """
    <div style="background-color:lightblue; padding:10px">
    <h3 style="color:black; text-align:center;">Machine Learning Model to predict Student Academic Performance </h3>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
     
    st.write("Enter the values below to predict the Academic performance of a particular student:")
    
    #taking input from  user
    col1,col2,col3, = st.columns(3)
    
    with col1:
        cgpa = st.number_input('CGPA', min_value=0.0, max_value=5.0, value=2.0)
        
    with col2:
        jamb_score = st.number_input('JAMB Score', min_value=50, max_value=400, value=200)
    
    with col3:
        waec_score = st.number_input('Olevel Aggregate score', min_value=0.0, max_value=5.0, value=2.0)

    #taking input from  user
    col4,col5 = st.columns(2)
    
    with col4:
        gender = st.selectbox('Gender:', ['Male', 'Female'])

    with col5:
        zone = st.selectbox("Zone:", ['South West', 'South East', 'South South', 'North Central', 'North East', 'North West'])


 
# Encoding gender and zone to numeric values
gender_mapping = {'Male': 0, 'Female': 1}
zone_mapping = {'South West': 5,  'South South': 4, 'South East': 3, 'North West': 2, 'North East': 1, 'North Central': 0}

# Convert selected values to numeric
gender_numeric = gender_mapping[gender]
zone_numeric = zone_mapping[zone]
     
    # code for prediction
    # Predict button
if st.button('Predict'):
    features = np.array([[gender_numeric,zone_numeric,cgpa, jamb_score, waec_score]])
    prediction = performance_model.predict(features)
    probability = performance_model.predict_proba(features)[0][1]

    if prediction == 1:
        st.markdown(f'<h4 style="color:green; background-color:#000; size:10px;">The model predicts that you <strong>are a student with an excellent performance</strong> with a probability of {probability:.2f}.</h4>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    else:
        st.markdown(f'<h4 style="color:orange; background-color:#000; size:10px">The model predicts that you <strong>are a student with an average performance</strong> with a probability of {1 - probability:.2f}.</h4>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    
     
 

    st.markdown(html_temp, unsafe_allow_html=True)
     

html_temp = """
    <div style="background-color:black; padding:10px"; color:white;>
    <h6 style="color:white; text-align:center;">&copy 2024 Created by: Odeyale Kehinde Musiliudeen </h6>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
        
