import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import pandas as pd

def app():
    st.title("Set Default")
    st.markdown("<hr style='margin-top: -1px; margin-bottom: 1px;'>", unsafe_allow_html=True)

    def connect_database():
        with sqlite3.connect("pakinson_admin.db") as db:
            c = db.cursor()
        choose_algo = c.execute('select * from model order by f1_score desc')#ดึงข้อมูลทุกคอลัมน์มา เรียงตาม f1 score
        model_algo = choose_algo.fetchall()
        c.close()
        return model_algo
    
    def default_UI():

        def set_Default():
            # The Key and Algorithm that user has been selected here
            default_key = default_model
            theAlgo = default_option

            conn = sqlite3.connect('pakinson_admin.db')
            cursor = conn.cursor()

            cursor.execute(f"UPDATE model SET m_default = ? WHERE algorithm = ?",("0",theAlgo))  # Assuming there's only one column "m_default" in the result
            cursor.execute(f"UPDATE model SET m_default = ? WHERE model_id = ? AND algorithm = ?",("1",str(default_key),theAlgo))
            conn.commit()
            conn.close()
            st.write("The model number:",default_model,'is now default of',theAlgo,"Algorithm")
            st.success('Setting Successfully')

        model_algo = connect_database()
        df = pd.DataFrame(model_algo, columns=['model_id', 'model_name', 'algorithm', 'hyperPM', 'model_path', 'dataset_path', 'admin_id', 'date', 'accuracy', 'precision', 'recall', 'f1_score',"default"])
        
        #Split specific table
        default_option = st.selectbox('Algorithm Selection', df['algorithm'].unique(), key='default_option')
        st.write(default_option,'has been selected.')

        filtered_data = df[df['algorithm'] == default_option]
        st.write(filtered_data)

        #Select Model <-------
        default_model = st.selectbox('Select Model id', filtered_data['model_id'].unique(), key='default_model')

        #display
        selected_name = filtered_data[filtered_data['model_id'] == default_model]['model_name'].values[0]
        st.write(default_option,"ID: ",default_model, "Name: ", selected_name, 'has been selected.')
        
        #SetDefault
        set_df = st.button('Set to default', on_click=set_Default)

    #These variable are for set_Default function
    default_UI()

