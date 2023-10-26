import streamlit as st
import sqlite3


def creds_entered():
    with sqlite3.connect("Parkinsons-Detector/pakinson_admin.db") as db:
        c = db.cursor()
    adminname = st.session_state['aduser'].strip()
    adminpass = st.session_state['adpasswd'].strip()
    user_check = ('select * from admin where username = ? and password = ?')
    c.execute(user_check, [(adminname),(adminpass)])
    result = c.fetchall()

    return result is not None

def Authenticate_user():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return None

    username_placeholder = st.empty()
    password_placeholder = st.empty()
    submit_button_placeholder = st.empty()

    username = username_placeholder.text_input(label='Username', value='', key='aduser')
    password = password_placeholder.text_input(label='Password', value='', key='adpasswd', type='password')
    submit_button = submit_button_placeholder.button("Login")
    
    if submit_button:
        if len(username) <= 0 or len(password) <= 0:
            st.warning('Please Enter Username/Password')
        else:
            result = creds_entered()
            if result:
                st.session_state['authenticated'] = True
                # Clear the placeholders to hide the form
                username_placeholder.empty()
                password_placeholder.empty()
                submit_button_placeholder.empty()
                return st.session_state['aduser']
            else:
                st.warning('Invalid Username or Password.')
                st.session_state['authenticated'] = False

def logout():
    st.session_state['authenticated'] = False

    



