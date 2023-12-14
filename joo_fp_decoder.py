# ===============================================================================================
# ATTENTION MECHANISM & DECODER
#
# Attention Mechanism enables the model to pay special attention to a particular part of text that relevant
# and give more weight to that specific part. It works similarly to how humans give more attention to a
# specific part when reading an article.
#
# Decoder works by decoding the vectors that contain information about the content of the article
# from Encoder and returning words that in the end will assemble a title.
# ===============================================================================================

# Import Libraries
from tensorflow.keras.layers import Embedding, Concatenate, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow import squeeze, concat
from tensorflow.linalg import matmul
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Define some hyperparameters
hidden_size = 200
embedding_dim = 150
batch_size = 128
epochs = 10
dropout_rate = 0.4
max_len_content = 500

# Load the tokenizer
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

# Attention Mechanism
class Attention(Model):
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs):
        encoder_hiddens, decoder_hidden = inputs

        # Calculate the attention score (dot product)
        attention_scores = matmul(decoder_hidden, encoder_hiddens, transpose_b=True)

        # Convert the attention scores to probability distribution using softmax
        attention_prob_distributions = softmax(attention_scores, axis=-1)

        # Get the context vector
        context_vector = matmul(attention_prob_distributions, encoder_hiddens)

        # Output the results
        return context_vector


# Define the decoder
class Decoder(Model):
    def __init__(self, input_dim, embedding_dim, hidden_size, dropout_rate):
        super(Decoder, self).__init__()

        # Initialize Embedding, LSTM, Attention, and Dense layer
        self.decoder_embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, name='DecoderEmbedding')
        self.decoder_lstm1 = LSTM(units=hidden_size*2, return_sequences=True, return_state=True, dropout=dropout_rate, name='DecoderLSTM')
        self.attention = Attention()
        self.attentional_hidden_state_layer = Dense(units=hidden_size, activation='tanh', name='AttentionalHiddenStateLayer')
        self.decoder_dense = Dense(units=input_size_title, name='DecoderDense')

    def call(self, decoder_inputs):
        decoder_input, encoder_hiddens, encoder_final_states = decoder_inputs

        # Apply the embedding
        decoder_embedding_output = self.decoder_embedding(decoder_input)

        # Apply the LSTM on the embedding result
        decoder_lstm1_output, decoder_lstm1_hidden, decoder_lstm1_cell = self.decoder_lstm1(decoder_embedding_output,
                                                                                            initial_state=encoder_final_states)

        # Get the context vector from attention mechanism
        context_vector = self.attention([encoder_hiddens, decoder_lstm1_output])

        # Concatenate the decoder's hidden state with context vector from attention mechanism
        concatenated_do_cv = concat([squeeze(decoder_lstm1_output, 1), squeeze(context_vector, 1)], -1)

        # Apply the tanh function
        decoder_output_temporary = self.attentional_hidden_state_layer(concatenated_do_cv)

        # Get the logits
        logits = self.decoder_dense(decoder_output_temporary)

        # Return the results
        return logits, decoder_lstm1_hidden, decoder_lstm1_cell