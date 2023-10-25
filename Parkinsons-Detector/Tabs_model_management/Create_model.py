import streamlit as st
from web_functions import load_data, train_model
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import joblib
import sqlite3
import datetime
import time


def app():
    df, X, y = load_data()
    st.title("Create Model")
    Model_title = st.text_input("Model Name")
    st.selectbox('Select Target', [y.name])
    st.multiselect('Select Feature',list(X.columns),default=list(X.columns))
    receive_algo = st.selectbox('Select Algorithm',('Random Forest', 'Support vector machine', 'Decision Tree', 'K nearest neighbor', "Naive Bayes"))
    fold = st.text_input("Input K (For K-Fold cross validation )")
    algorithm = None        

    def createconnection():
        global conn,cursor
        conn = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
        cursor = conn.cursor()

    def checking_input(Model_title, fold):
        createconnection()
        #check if name is already exist.
        check1 = False
        check2 = False
        cursor.execute(f"SELECT model_name FROM model WHERE model_name='{Model_title}'")
        nameCheck = cursor.fetchone()
        if nameCheck or len(Model_title) == 0:
            st.error("The name already exists.\nPlease enter a different name.")
        else:
             check1 = True
        if len(fold) == 0 or fold.isnumeric() is not True:
            st.error("Please enter K-fold number")
        else:
            check2 = True

        if check1 == True and check2 == True:
            return True
            
        

    def modelSaving():
        createconnection()
        current_datetime = datetime.datetime.now()
        # Create a date object containing only year, month, and day
        get_date = current_datetime.date()
        cursor.execute("SELECT model_id FROM model ORDER BY CAST(model_id AS INTEGER) ASC")
        get_cm_id = cursor.fetchall()
        count_cm = int(get_cm_id[-1][0])+1

        name = "model"+str(count_cm)
        joblib.dump(model, name+ '.joblib')
        model_path = name+ '.joblib'
    
        cursor.execute(f"INSERT INTO model (model_id, model_name, algorithm, hyperPM, model_path, dataset_path, admin_id, date, accuracy, precision, recall, f1_score) VALUES ('{str(count_cm)}', '{Model_title}', '{receive_algo}', '{fold}', '{model_path}','{'Parkinson.csv'}', '{'1'}', '{get_date}', '{accuracy}', '{precision}', '{recall}', '{f1}')")
        conn.commit()
        success_message = st.success("The model has been saved.")
                
    
    if st.button("Train Model"):
        if receive_algo == "K nearest neighbor":
            algorithm = KNeighborsClassifier(n_neighbors=45, weights = 'distance')
        elif receive_algo == "Random Forest":
            algorithm = RandomForestClassifier()
        elif receive_algo == "Decision Tree":
            algorithm = DecisionTreeClassifier()
        elif receive_algo == "Support vector machine":
            algorithm = LinearSVC()
        elif receive_algo == "Naive Bayes":
            algorithm = GaussianNB()


        if checking_input(Model_title, fold):
            accuracy, precision, recall, f1, model = train_model(algorithm ,int(fold), X, y)
            accuracy, precision, recall, f1 = np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)
            matrix = {
                'Accuracy': np.mean(accuracy),
                'Precision': np.mean(precision),
                'Recall': np.mean(recall),
                'F1-score': np.mean(f1)}     
            
            df_matrix  = pd.DataFrame([matrix])
            st.table(df_matrix)
            st.button("Save Model", on_click=modelSaving)
        
        
            
        
            
    
