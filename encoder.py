import tensorflow as tf

from encoder_unit import EncoderUnit

class Encoder(tf.keras.Model):
    
    def __init__(self, num_heads, enc_layers, d_model, filters, filter_size):
        super(Encoder, self).__init__()
        
        self.num_heads  = num_heads
        self.enc_layers = enc_layers
        self.d_model = d_model
        self.filters  = filters
        self.filter_size = filter_size
        self.encoder_units  = []
    
        for layer in range(enc_layers):
            encoder_unit = EncoderUnit(num_heads, d_model, filters, filter_size)
            self.encoder_units.append(encoder_unit)
    

    def call(self, values, training=True):
        
        for layer in range(self.enc_layers):
            values = self.encoder_units[layer](values, training)
        # print("Encoder : ", values.shape)
        return values
