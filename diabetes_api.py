# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:14:23 2025

@author: USER
"""

from fastapi import FastAPI
from pydanic import BaseModel
import pickle
import json

app=FastAPI()


class model_input(BaseModel):
    
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int
    
    
#loading the saved model
diabetes_model=pickle.load(open('trained_model.sav','rb'))

@app.get('/diabetes_prediction')
def diabetes_pred(input_parameters:model_input):
    
    input_data=input_parameters.json()
    input_dictionary=json.loads(input_data)
    
    preg=input_dictionary['Pregnancies']
    glu=input_dictionary['Glucose']
    bp=input_dictionary['BloodPressure']
    skin=input_dictionary['SkinThickness']
    insulin=input_dictionary['Insulin']
    bmi=input_dictionary['BMI']
    dpf=input_dictionary['DiabetesPedigreeFunction']
    age=input_dictionary['Age']
    
    
    input_list=[preg,glu,bp,skin,insulin,bmi,dpf,age]
    
    prediction=diabetes_model.predict([input_list])
    
    if prediction[0]==0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
      
     
    
    