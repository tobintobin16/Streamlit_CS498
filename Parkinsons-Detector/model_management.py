"""This is the main module to run the app"""

# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from web_functions import load_data

# Import pages
from Tabs_model_management import Create_model, Edit_model,Set_default
from streamlit_option_menu import option_menu
from Login import Authenticate_user, logout

st.set_page_config(
    page_title='Management System',
    page_icon='raised_hand_with_fingers_splayed',
    layout='wide',
    initial_sidebar_state='auto'
)

username = Authenticate_user()

if st.session_state.authenticated:
    # st.markdown(f'<p style="color:green">{st.session_state['aduser']}</p>', unsafe_allow_html=True)
    Tabs = {
    "Create Model": Create_model,
    "Set Default": Set_default,
    "Edit Model": Edit_model
    
}

    with st.sidebar:
        page = option_menu(menu_title="Main Menu",options=list(Tabs.keys()))
        st.sidebar.markdown("---")
        logout_button = st.button('Logout', on_click=logout)  # Display a logout button
        
    Tabs[page].app()
