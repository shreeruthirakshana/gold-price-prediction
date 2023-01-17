# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 23:48:22 2023

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st
st.markdown("""
    <style>
        .stApp {
        background: url("https://img.freepik.com/premium-vector/business-candle-stick-graph-chart-stock-market-investment-trading_41981-1325.jpg?w=2000");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)
loaded_model=pickle.load(open('C:/Users/Admin/anaconda3/Lib/site-packages/pickle_mixin/test/gold_model.sav','rb'))

def goldprediction(input_data):
    
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    return prediction

def main():
    
    st.title('GOLD PRICE PREDICTION WEP APP')
    spx=st.text_input('ENTER SPX VALUE:')
    uso=st.text_input('ENTER USO VALUE:')
    slv=st.text_input('ENTER SILVER PRICE:')
    eur=st.text_input('ENTER EUR / USD VALUE:')
    
    goldpredictions = ''
    
    if st.button('PREDICT THE VALUE:'):
        goldpredictions=goldprediction([spx,uso,slv,eur])
        
    st.success(goldpredictions)
    
    
if __name__=='__main__':
    main()