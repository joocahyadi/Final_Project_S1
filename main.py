# ===============================================================================================
# MAIN
#
# This script is focused on loading the model, accepting user input, and generating the title for the specific
# user input. The API for serving the whole ML model and running inference, as well as returning the
# prediction is available in the end.
# ===============================================================================================

# Import libraries
from fastapi import FastAPI
import numpy as np
import json
from fastapi.responses import JSONResponse

from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from joo_fp_encoder import Encoder
from joo_fp_decoder import Attention, Decoder

# Define some hyperparameters
hidden_size = 200
embedding_dim = 150
batch_size = 128
epochs = 10
dropout_rate = 0.4
max_len_content = 500


# Load model
## Load tokenizer
### Article tokenizer
with open(r'D:\Kuliah S1 Matematika\Tugas Akhir Joshia\Model\Tokenizer Final\article_tokenizer_v6_subclassing_new_data_v3_with_attention.json') as f:
    data = json.load(f)
    article_tokenizer = tokenizer_from_json(data)

### Title tokenizer
with open(r'D:\Kuliah S1 Matematika\Tugas Akhir Joshia\Model\Tokenizer Final\title_tokenizer_v6_subclassing_new_data_v3_with_attention.json') as f:
    data = json.load(f)
    title_tokenizer = tokenizer_from_json(data)

# Add some hyperparameters
input_size_article = len(article_tokenizer.word_index)+1
input_size_title = len(title_tokenizer.word_index)+1


## Deep learning model
### Encoder
encoder = Encoder(input_dim=input_size_article, hidden_size=hidden_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
decoder = Decoder(input_dim=input_size_title, hidden_size=hidden_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)

encoder.load_weights('D:\Kuliah S1 Matematika\Tugas Akhir Joshia\Model\Model Final\encoder_weights_v6_subclassing_new_data_v3_with_attention')
decoder.load_weights('D:\Kuliah S1 Matematika\Tugas Akhir Joshia\Model\Model Final\decoder_weights_v6_subclassing_new_data_v3_with_attention')


# Prediction
## Define the function
def generate_title_with_attention(input_text, article_tokenizer=article_tokenizer, title_tokenizer=title_tokenizer, encoder=encoder, decoder=decoder, max_title_length=15):

    # Tokenize the input text
    input_sequence = article_tokenizer.texts_to_sequences([input_text])

    # Pad the sequence
    input_sequence = pad_sequences(input_sequence, maxlen=max_len_content, padding='post')

    # Initialize the first word for decoder
    current_word = '<start>'

    # Initialize a list to contain all the words
    title_container = []

    # Get the final states (hidden and cell) from encoder
    encoder_output, hidden_state, cell_state = encoder.predict(input_sequence)

    # Loop
    while len(title_container) < max_title_length:

        # Initialize a container vector to contain words that will become the inputs for decoder
        input_word = np.zeros((1,1))

        # Get the index of the current word and save it to the container (input_word)
        input_word[0,0] = title_tokenizer.word_index[current_word]

        # Get the result from decoder
        decoder_output, hidden_state, cell_state = decoder.predict([input_word, encoder_output, (hidden_state, cell_state)], verbose=None)

        # Get the index (position) of a word with the highest index
        resulting_word_index = np.argmax(decoder_output[0])

        # Get the word itself
        resulting_word = title_tokenizer.index_word[resulting_word_index]

        # Change the content of current_word to the resulting_word, and use it for decoder's next iteration
        current_word = resulting_word

        # If the word that decoder generates is '<end>', then breaks the loop
        if current_word == '<end>':
            break

        # Append the resulting_word to the container
        title_container.append(resulting_word)

    # Return the title
    return ' '.join(title_container)


# Inference API
app = FastAPI()

app.max_request_size = 100 * 1024 * 1024

@app.post('/predict')
def predict(input_text: str):

    # Get the prediction and make it in json format
    response_data = {'predicted_title': generate_title_with_attention(input_text)}

    # Return the json-style response
    return JSONResponse(content=response_data)
