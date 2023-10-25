import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import pandas as pd

def app():
    st.title("Edit model")

    def connect_database():
            with sqlite3.connect("Parkinsons-Detector/pakinson_admin.db") as db:
                c = db.cursor()
            choose_algo = c.execute('select * from model order by f1_score desc')#ดึงข้อมูลทุกคอลัมน์มา เรียงตาม f1 score
            model_algo = choose_algo.fetchall()
            c.close()
            return model_algo
    
    def edit_UI():
        model_algo = connect_database()

        def change_model_name():
            conn = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
            c = conn.cursor()

            c.execute("UPDATE model SET model_name = ? WHERE model_id = ?", (changename, selected_model))
            conn.commit()
            conn.close()
            st.success('The model name has been changed')

        df = pd.DataFrame(model_algo, columns=['model_id', 'model_name', 'algorithm', 'hyperPM', 'model_path', 'dataset_path', 'admin_id', 'date', 'accuracy', 'precision', 'recall', 'f1_score',"default"])
        selected_option = st.selectbox('Algorithm Selection', df['algorithm'].unique(), key='edit_option')
        st.write(selected_option,'has been selected.')
        #ChangeName
        filtered_data = df[df['algorithm'] == selected_option]
        st.write(filtered_data)

        #Select Model <-------
        selected_model = st.selectbox('Select Model id', filtered_data['model_id'].unique(), key='edit_model')

        #display
        selected_name = filtered_data[filtered_data['model_id'] == selected_model]['model_name'].values[0]

        changename_placeholder = st.empty()
        changename = changename_placeholder.text_input(label='Change model mame', value=selected_name, key='cn')
        cha_nm = st.button('Edit Model', on_click=change_model_name)
        
        return selected_model, selected_option, selected_name

    def delete_UI():
        def deleteconfirm():
            connd = sqlite3.connect('Parkinsons-Detector/pakinson_admin.db')
            cursord = connd.cursor()

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
            Ybtn = st.button('Confirm' , on_click= deleteconfirm)
            Nbtn = st.button('Cancle')
            ChangeButtonColour('yes', '#FF0000')
    

    def ChangeButtonColour(wgt_txt, wch_hex_colour = '12px'):
        htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.color ='""" + wch_hex_colour + """'; } }</script>  """


        htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
        components.html(f"{htmlstr}", height=0, width=0)

    selected_model, selected_option, selected_name = edit_UI()
    delete_UI()
