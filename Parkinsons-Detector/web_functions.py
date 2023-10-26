"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import joblib
import os


@st.cache_data
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Parkinsons-Detector/Parkinson.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns = {"MDVP:Fo(Hz)": "AVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Fhi(Hz)": "MAVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Flo(Hz)": "MIVFF",}, inplace = True)

    # Perform feature and target split
    X  = df.drop(columns=['status','name'])
    y = df['status']

    return df, X, y

def train_model(algorithm, K, X, y):
    """This function trains the model and returns the model scores"""    
    def cross_validation(algorithm, _X, _y, _cv):
        _scoring = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(estimator=algorithm,
                                X=_X,
                                y=_y,
                                cv=_cv,
                                scoring=_scoring,
                                return_train_score=True,
                                return_estimator=True
                                )
        
        return results

    result = cross_validation(algorithm,X,y,K)

    accuracy =  result['test_accuracy']
    precision = result['test_precision']
    recall = result['test_recall']
    f1 = result['test_f1']
    best_f1 = list(result['test_f1']).index(max(result['test_f1']))
    model=result["estimator"][best_f1]

    return accuracy, precision, recall, f1, model

# def predict(X, y, features):
#     # Get model and model score

#     a,b,c,d, model = train_model(DecisionTreeClassifier(), 10, X, y)
#     # Predict the value
#     prediction = model.predict([features])

#     return prediction, np.max(a)

def predict(features, algorithm):
    print(features)
    print("*"*30)
    print(algorithm)
    # Get model and model score
    conn = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
    cursor = conn.cursor()
    exc = cursor.execute("SELECT model_path, accuracy FROM model WHERE m_default=? AND algorithm=?", ("1",algorithm))
    model = exc.fetchall()

    file_path = "Parkinsons-Detector/"+model[0][0]
    # Deserialize the model from the file
    # model_path = model[0][0]
    loaded_model = joblib.load(file_path)
    # Predict the value
    prediction = loaded_model.predict([features])

    return prediction, model[0][1]
