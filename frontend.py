import streamlit as st
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json


# Load files 

with open('gb_grid_best.pkl', 'rb') as file_1:
  gb_grid_best = joblib.load(file_1)

with open('numerical_features.txt', 'r') as file_2:
  numerical_features = json.load(file_2)

def run():
    st.title('Plant Nutrition Prediciton')
    st.write('by :  Muhammad Julizar')


    with st.form(key='form_parameters'):
        v1 = st.number_input('v1', min_value=220, max_value=680, value=220, step=25)
        v2 = st.number_input('v1', min_value=170, max_value=430, value=170, step=25)
        v3 = st.number_input('v1', min_value=340, max_value=730, value=340, step=25)
        v4 = st.number_input('v1', min_value=300, max_value=450, value=300, step=25)
        v5 = st.number_input('v1', min_value=350, max_value=750, value=350, step=25)
        v6 = st.number_input('v1', min_value=150, max_value=450, value=150, step=25)
        v7 = st.number_input('v1', min_value=500, max_value=900, value=500, step=25)
        v8 = st.number_input('v1', min_value=3700, max_value=5100, value=3700, step=25)
        sample_type = st.selectbox('sample_type', ('Lab 1', 'Lab 2'), index=1)
        st.markdown('---')

        submitted = st.form_submit_button('Predict')




    data_inf={
            'v1'   : v1,
            'v2'   : v2,
            'v3'   : v3,
            'v4'   : v4,
            'v5'   : v5,
            'v6'   : v6,
            'v7'   : v7,
            'v8'   : v8,
            'sample_type': sample_type
            }


    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:       
        y_pred = gb_grid_best.predict(data_inf)

        st.write('# target:', str(int(y_pred)))
        

if __name__ == '__main__':
    run()