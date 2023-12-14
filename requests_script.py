# ===============================================================================================
# REQUESTS SCRIPT
#
# This script is used for building the UI where user can directly input the text or article.
# After that, the text or article will be send to the ML model via API in main.py script.
# ===============================================================================================


# Import Libraries
import requests
import json

# Main part of this script
if __name__ == "__main__":

    # Define the url
    url = "http://127.0.0.1:8000/predict"

    # Get the user input text
    user_text = str(input('Enter the text/article here: '))

    # Define the input data in JSON format
    input_data = {"input_text": user_text}

    # Post the input_data to the ML model and request the prediction
    response = requests.post(url=url, params=input_data)

    # Parse the json formatted output
    result = json.loads(response.content)

    # Print the status code and result
    print('Status code:', response.status_code)
    print()
    print('Title predicted by the model:')
    print(result['predicted_title'])
