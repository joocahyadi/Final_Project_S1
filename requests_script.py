# ===============================================================================================
# REQUESTS SCRIPT
#
# This script is used for building the UI where user can directly input the text or article.
# After that, the text or article will be send to the ML model via API in main.py script.
# ===============================================================================================


# Import Libraries
import requests
import json
import streamlit as st

# Function
def give_prediction(input_user):

    # Define the url
    url = "http://127.0.0.1:8000/predict"

    # Define the input data in JSON format
    input_data = {"input_text": input_user}

    # Post the input_data to the ML model and request the prediction
    response = requests.post(url=url, params=input_data)

    # Return the result (predicted title to the user)
    result = json.loads(response.content)

    return result


# Main part of this script
if __name__ == "__main__":
    # UI Part
    # Title
    st.title("Title Recommendation for Indonesian News Article :newspaper:")

    # Introduction
    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader('Hi and Welcome!')
    st.write(
        """
        This web application will recommend the appropriate title for an Indonesian News Article that you have.
        Simply put your article in the box below and press "Predict for Me!" button.
        
        Wait a few seconds and voila! The result will pop out of nowhere! :tada:
        """
    )

    # Create the input field
    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader('Give me your article!')
    user_text = st.text_area(label='You can put it here:')

    # Create the button for prediction
    if st.button('Predict for Me!'):
        pred = give_prediction(user_text)

        # Run the give_prediction function
        st.subheader(f"Your predicted title:")
        st.write(pred['predicted_title'])
