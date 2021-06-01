import tensorflow as tf

from .utility import positional_encoding
from .decoder_layer import DecoderLayer

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, max_position_encoding):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = tf.keras.layers.Conv2D(d_model, filter_size, 
                                                padding='same', activation='relu')
        self.pos_encoding = positional_encoding(max_position_encoding, image_shape, d_model)
        
        self.dec_layers  = [DecoderLayer(d_model, num_heads, dff, filter_size)
                            for _ in range(num_layers)]
    

    def call(self, x, enc_output, training, look_ahead_mask):

        # enc_output.shape = (batch_size, input_seq_len, rows, cols, depth)
        
        seq_len = x.shape[1]
        attention_weights = {}
        
        # image embedding and position encoding
        x = self.embedding(x) # (batch_size, target_seq_len, rows, cols, depth)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :, :, :]
        
        for layer in range(self.num_layers):
            x, block1, block2 = self.dec_layers[layer](x, enc_output,
                                                       training, look_ahead_mask)

            attention_weights[f'decoder_layer{layer+1}_block1'] = block1
            attention_weights[f'decoder_layer{layer+1}_block2'] = block2

        # x.shape = (batch_size, target_seq_len, rows, cols, d_model)
        return x, attention_weights

