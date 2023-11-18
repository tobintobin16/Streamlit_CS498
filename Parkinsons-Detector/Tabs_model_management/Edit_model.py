import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import pandas as pd
from web_functions import train_model , load_data, sampling_function
import numpy as np

# Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

import os
import joblib

def app():
    st.title("Edit model")
    st.markdown("<hr style='margin-top: -1px; margin-bottom: 1px;'>", unsafe_allow_html=True)

    def connect_database():
            with sqlite3.connect("Parkinsons-Detector/pakinson_admin.db") as db:
                c = db.cursor()
            choose_algo = c.execute('select * from model order by f1_score desc')#ดึงข้อมูลทุกคอลัมน์มา เรียงตาม f1 score
            model_algo = choose_algo.fetchall()
            c.close()
            return model_algo
    
    def algoChecking():
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
        elif receive_algo == "Logistic Regression":
            algorithm = LogisticRegression()
        elif receive_algo == "Gradient Boosting":
            algorithm = HistGradientBoostingClassifier(max_iter=100)
        elif receive_algo == "XGBClassifier":
            algorithm = XGBClassifier()
        return algorithm
    
    def edit_UI():
        model_algo = connect_database()

        df = pd.DataFrame(model_algo, columns=['model_id', 'model_name', 'algorithm', 'k_fold', 'sampling', 'model_path', 'dataset_path', 'admin_id', 'date', 'accuracy', 'precision', 'recall', 'f1_score',"default"])
        selected_option = st.selectbox('Algorithm Selection', df['algorithm'].unique(), key='edit_option')
        st.write(selected_option,'has been selected.')
        #ChangeName
        filtered_data = df[df['algorithm'] == selected_option]
        st.write(filtered_data)

        #Select Model <-------
        selected_model = st.selectbox('Select Model id', filtered_data['model_id'].unique(), key='edit_model')

        #get needed value
        global selected_name, selected_fold, selected_fold, receive_algo, selected_dataset, selected_sampling, selected_path
        selected_name = str(filtered_data[filtered_data['model_id'] == selected_model]['model_name'].values[0])
        selected_fold = filtered_data[filtered_data['model_id'] == selected_model]['k_fold'].values[0]
        receive_algo = filtered_data[filtered_data['model_id'] == selected_model]['algorithm'].values[0]
        selected_dataset = filtered_data[filtered_data['model_id'] == selected_model]['dataset_path'].values[0]
        selected_sampling = filtered_data[filtered_data['model_id'] == selected_model]['sampling'].values[0]
        selected_path = filtered_data[filtered_data['model_id'] == selected_model]['model_path'].values[0]

        #display
        global change_fold, changename
        changename = st.text_input(label='Change model mame', value=selected_name, key='cn')
        change_fold = st.text_input(label='Input new fold', value=selected_fold, key='cf')
        cha_nm = st.button('Edit Model')
        if cha_nm:
            change_model_fold()
        
        return selected_option, selected_model
    
    def change_model_fold():
        checked = False

        def non_impact():
            conn = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
            c = conn.cursor()
            c.execute("UPDATE model SET model_name=? WHERE model_id = ?", (changename, selected_model))
            conn.commit()
            conn.close()
            st.success('Successfully')

        def model_impact():
            conn = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
            c = conn.cursor()

            if checked:
                the_model =  os.path.join("Parkinsons-Detector/" + "models" +"/"+ str(selected_path))
                # Check if the file exists
                if os.path.exists(the_model):
                    # If the file exists, remove it
                    os.remove(the_model)

                # Save the new model
                joblib.dump(model, the_model)
            
            c.execute("UPDATE model SET model_name=?, k_fold=?, accuracy=?, precision=?, recall=?, f1_score=? WHERE model_id = ?", (changename, change_fold, accuracy, precision, recall, f1, selected_model))
            conn.commit()
            conn.close()
            st.success('Successfully')

        if int(selected_fold) is not int(change_fold): 
            df, X, y = load_data(selected_dataset)
            if selected_sampling == 'Over sampling' or selected_sampling == 'Under sampling':
                X, y = sampling_function(selected_sampling, X, y)

            algorithm = algoChecking()

            accuracy, precision, recall, f1, model = train_model(algorithm ,int(change_fold), X.values, y.values)
            accuracy, precision, recall, f1 = np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)
            matrix = {
                'Accuracy': np.mean(accuracy),
                'Precision': np.mean(precision),
                'Recall': np.mean(recall),
                'F1-score': np.mean(f1)}
            
            df_matrix  = pd.DataFrame([matrix])
            st.table(df_matrix)  
            checked = True          
            st.button('Save changed', on_click=model_impact)

        # print(selected_name, changename)
        if checked != True and selected_name != changename:
            st.button('Save changed', on_click=non_impact)
            
    def delete_UI():
        def deleteconfirm():
            connd = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
            cursord = connd.cursor()

            d_model = os.path.join("Parkinsons-Detector/" + "models" +"/"+ str(selected_path))
            # Check if the file exists
            if os.path.exists(d_model):
                # If the file exists, remove it
                os.remove(d_model)

            cursord.execute('DELETE FROM model WHERE model_id=?', (int(selected_model),))
            connd.commit()
            connd.close()
            st.success('The model has been deteted')

        # Add custom CSS to style the button
        delete_button = st.button("Delete Model")
        ChangeButtonColour('Delete Model', '#FF0000')

        if delete_button:
            st.warning('Delete Confirmation',icon="⚠️")
            st.write("Algorithm: ",selected_option,"ID: ",selected_model, "Name: ", selected_name)  
            Ybtn = st.button('Confirm', on_click= deleteconfirm)
            Nbtn = st.button('Cancle')
            ChangeButtonColour('yes', '#FF0000')
    

    def ChangeButtonColour(wgt_txt, wch_hex_colour = '12px'):
        htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.color ='""" + wch_hex_colour + """'; } }</script>  """


        htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
        components.html(f"{htmlstr}", height=0, width=0)

    selected_option, selected_model =  edit_UI()
    delete_UI()
