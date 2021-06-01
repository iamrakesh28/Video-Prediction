import tensorflow as tf

from .utility import positional_encoding
from .encoder_layer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, max_position_encoding):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = tf.keras.layers.Conv2D(d_model, filter_size, 
                                                padding='same', activation='relu')
        self.pos_encoding = positional_encoding(max_position_encoding, image_shape, d_model)
        
        self.enc_layers  = [EncoderLayer(d_model, num_heads, dff, filter_size)
                            for _ in range(num_layers)]
    

    def call(self, x, training, mask):

        # x.shape = (batch_size, seq_len, rows, cols, depth)
        seq_len = x.shape[1]
        
        # image embedding and position encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :, :, :]
        
        for layer in range(self.num_layers):
            x = self.enc_layers[layer](x, training, mask)

        return x # (batch_size, seq_len, rows, cols, d_model)
