"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict, load_data

import sqlite3


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                Input Data for the Prediction of Parkinson's disease.
            </p>
        """, unsafe_allow_html=True)
   
    with st.expander("View attribute details"):
        st.markdown("""MDVP:Fo(Hz) - Average vocal fundamental frequency\n
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency\n
MDVP:Flo(Hz) - Minimum vocal fundamental frequency\n
MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several
measures of variation in fundamental frequency\n
MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude\n
NHR,HNR - Two measures of ratio of noise to tonal components in the voice\n
status - Health status of the subject (one) - Parkinson's, (zero) - healthy\n
RPDE,D2 - Two nonlinear dynamical complexity measures\n
DFA - Signal fractal scaling exponent\n
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation""")
    algorithm = st.selectbox('Select Algorithm',('Random Forest', 'Support vector machine', 'Decision Tree', 'K nearest neighbor', 'Naive Bayes'))
    # Take feature input from the user
    # Add a subheader
    st.subheader("Input Values:")
    # Create a dictionary to hold the input field values and their corresponding labels
    input_fields = {
        "Average vocal fundamental frequency": "avff",
        "Jitter(%)-MDVP": "jitter(%)",
        "PPQ-MDVP": "PPQ_mdvp",
        "Shimmer(dB)-MDVP": "shim_dB",
        "APQ-MDVP": "APQ_mdvp",
        "HNR": "hnr",
        "spread1": "spread1",
        "PPE": "ppe",
        "Maximum vocal fundamental frequency": "mavff",
        "Jitter(Abs)-MDVP": "jitter_ABS",
        "DDP-Jitter": "DDP_jitter",
        "Shimmer-APQ3": "shimapq3",
        "Shimmer-DDA": "DDA_shim",
        "RPDE": "rpde",
        "spread2": "spread2",
        "Minimum vocal fundamental frequency": "mivff",
        "RAP-MDVP": "RAP_mdvp",
        "Shimmer-MDVP": "Shim_MDVP",
        "Shimmer-APQ5": "shimapq5",
        "NHR": "nhr",
        "DFA": "dfa",
        "D2": "d2",
    }

    # Create a layout with three columns
    col, col1, col2 = st.columns([1, 1, 1])
    # Create dictionaries to store input field values and their respective empty statuses
    input_values = {}
    input_empty = {}

    # Function to display text input fields and highlight empty fields
    def display_text_input(col, label, key):
        input_values[key] = col.text_input(label, key=key)
        input_empty[key] = not input_values[key]

    # Display text input fields in the columns
    display_text_input(col,"Average vocal fundamental frequency", "avff")
    display_text_input(col,"Jitter(%)-MDVP", "jitter(%)")
    display_text_input(col,"PPQ-MDVP", "PPQ_mdvp")
    display_text_input(col,"Shimmer(dB)-MDVP", "shim_dB")
    display_text_input(col,"APQ-MDVP", "APQ_mdvp")
    display_text_input(col,"HNR", "hnr")
    display_text_input(col,"spread1", "spread1")
    display_text_input(col,"PPE", "ppe")

    display_text_input(col1,"Maximum vocal fundamental frequency", "mavff")
    display_text_input(col1,"Jitter(Abs)-MDVP", "jitter_ABS")
    display_text_input(col1,"DDP-Jitter", "DDP_jitter")
    display_text_input(col1,"Shimmer-APQ3", "shimapq3")
    display_text_input(col1,"Shimmer-DDA", "DDA_shim")
    display_text_input(col1,"RPDE", "rpde")
    display_text_input(col1,"spread2", "spread2")

    display_text_input(col2,"Minimum vocal fundamental frequency", "mivff")
    display_text_input(col2,"RAP-MDVP", "RAP_mdvp")
    display_text_input(col2,"Shimmer-MDVP", "Shim_MDVP")
    display_text_input(col2,"Shimmer-APQ5", "shimapq5")
    display_text_input(col2,"NHR", "nhr")
    display_text_input(col2,"DFA", "dfa")
    display_text_input(col2,"D2", "d2")

    #Check if dataset has Gender and Age
    conn = sqlite3.connect('pakinson_admin.db')
    cursor = conn.cursor()
    exc = cursor.execute("SELECT dataset_path FROM model WHERE m_default=? AND algorithm=?", ("1",algorithm))
    checked = exc.fetchall()
    
    if checked[0][0] == "Parkinson_V2.csv":
        df2, X2, y2 = load_data(checked[0][0])
        if 'Gender' and 'Age' in X2.columns:
            display_text_input(col1,"Gender","gender")
            display_text_input(col2,"Age","age")
        else:
            if 'gender' and 'age' in input_values:
                input_values.pop(['gender','age'])
    #-----------------------------------

    # Create a button to predict
    if st.button("Predict"):
        empty_fields = [label for label, key in input_empty.items() if key]
        if empty_fields:
            st.warning(f"Please fill in the following fields: {',  '.join(empty_fields)}")
        else:
            # Get prediction and model score
            integer_list = [float(x) for x in list(input_values.values())]
            prediction, score = predict(integer_list, algorithm)
            # st.success("Predicted Sucessfully")

            # Print the output according to the prediction
            if (prediction == 1):
                st.warning("You get: "+str(prediction)+ "The person either has Parkison's disease or prone to get Parkinson's disease")
            else:
                st.info("The person is safe from Parkinson's disease")

            # Print teh score of the model 
            st.write("The model used is trusted by doctor and has an accuracy of ", (score*100),"%")