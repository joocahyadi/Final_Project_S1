# ===============================================================================================
# ENCODER
#
# The Encoder is used to accept the article from user, extract the article's information,
# and return it as some vectors. The Encoder consists of Embedding and Bidirectional LSTM layers.
# ===============================================================================================

# Import the Libraries
from tensorflow.keras.layers import Embedding, Concatenate, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model, load_model


# Define some hyperparameters
hidden_size = 200
embedding_dim = 150
batch_size = 128
epochs = 10
dropout_rate = 0.4
max_len_content = 500


# Define the encoder
class Encoder(Model):
    def __init__(self, input_dim, hidden_size, embedding_dim, dropout_rate):
        super(Encoder, self).__init__()

        # Initialize Embedding, LSTM 1, LSTM 2, and Concatenate Layer
        self.encoder_embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, name='EncoderEmbedding')
        self.encoder_lstm1 = Bidirectional(LSTM(units=hidden_size, return_sequences=True, return_state=True, dropout=dropout_rate), name='BidirectionalLSTM1')
        self.encoder_lstm2 = Bidirectional(LSTM(units=hidden_size, return_sequences=True, return_state=True, dropout=dropout_rate), name='BidirectionalLSTM2')
        self.concatenate_layer = Concatenate()

    def call(self, encoder_input):

        # Apply the embedding
        encoder_embedding_output = self.encoder_embedding(encoder_input)

        # Apply the LSTM 1 and LSTM 2 to the embedding results
        encoder_lstm1_output, encoder_lstm1_hidden_forward, encoder_lstm1_hidden_backward, encoder_lstm1_cell_forward, encoder_lstm1_cell_backward = self.encoder_lstm1(
            encoder_embedding_output)
        encoder_lstm2_output, encoder_lstm2_hidden_forward, encoder_lstm2_hidden_backward, encoder_lstm2_cell_forward, encoder_lstm2_cell_backward = self.encoder_lstm2(
            encoder_lstm1_output)

        # Concatenate the result from forward and backward phase
        encoder_final_hidden = self.concatenate_layer([encoder_lstm2_hidden_forward, encoder_lstm2_hidden_backward])
        encoder_final_cell = self.concatenate_layer([encoder_lstm2_cell_forward, encoder_lstm2_cell_backward])

        # Output the result
        return encoder_lstm2_output, encoder_final_hidden, encoder_final_cell