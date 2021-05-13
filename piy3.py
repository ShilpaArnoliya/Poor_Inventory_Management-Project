# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:28:01 2021

@author: hp
"""

import pickle
#from flasgger import Swagger
import streamlit as st 
#app=Flask(__name__)
#Swagger(app)

pickle_in = open("reg3.pkl","rb")
model=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Product,sales):
    
    prediction = model.predict([[Product,sales]])
    print(prediction)
    return prediction



def main():
    st.title("Inventory Management")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Inventory Management ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Product = st.text_input("Product type","Type Here")
    #Revenue = st.text_input("Revenue","Type Here")
    #CPU = st.text_input("CPU","Type Here")
    #TD = st.text_input("TD","Type Here")
    #PA = st.text_input("Promotion applied","Type Here")
    #GH = st.text_input("Generic Holiday","Type Here")
    #EH = st.text_input("Education Holiday","Type Here")
    #DW = st.text_input("DayOfWeek","Type Here",)
    sales = st.text_input("sales","Type Here",type='default')
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Product,sales)
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()