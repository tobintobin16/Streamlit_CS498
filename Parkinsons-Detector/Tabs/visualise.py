"""This modules contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
import streamlit as st
# from mlxtend.plotting import plot_confusion_matrix


# Import necessary functions from web_functions
from web_functions import train_model

def app(df, X, y):
    """This function create the visualisation page"""
    
    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise the Parkinson's Prediction")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize = (10, 6))
        ax = sns.heatmap(df.iloc[:, 1:].corr(), annot = True)   # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim()                             # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5)                    # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot(fig)

    if st.checkbox("Show Scatter Plot"):
        
        figure, axis = plt.subplots(2, 2,figsize=(15,10))

        sns.scatterplot(ax=axis[0,0],data=df,x='AVFF',y='MAVFF',hue='status')
        axis[0, 0].set_title("Oversampling Minority Scatter")
  
        sns.countplot(ax=axis[0, 1],x="status", data=df)
        axis[0, 1].set_title("Oversampling Minority Count")
  
        sns.scatterplot(ax=axis[1, 0],data=df,x='AVFF',y='MAVFF',hue='status')
        axis[1, 0].set_title("Undersampling Majority Scatter")
  
        sns.countplot(ax=axis[1, 1],x="status", data=df)
        axis[1, 1].set_title("Undersampling Majority Count")
        st.pyplot()

    if st.checkbox("Display Boxplot"):
        fig, ax = plt.subplots(figsize=(15,5))
        df.boxplot(['AVFF', 'MAVFF', 'MIVFF','HNR'],ax=ax)
        st.pyplot()

    if st.checkbox("Show Sample Results"):
        safe = (df['status'] == 0).sum()
        prone = (df['status'] == 1).sum()
        data = [safe,prone]
        labels = ['Safe', 'Prone']
        colors = sns.color_palette('pastel')[0:7]
        plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        st.pyplot()
        
    if st.checkbox("Number of Patients Effected"):
        plt.figure(figsize=(10, 6))
        df.status.hist()
        plt.xlabel('status')
        plt.ylabel('Frequencies')
        st.pyplot()
        # The dataset has high number of patients effected with Parkinson's disease.

    
    
