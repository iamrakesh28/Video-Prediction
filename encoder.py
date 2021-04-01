import tensorflow as tf

from encoder_unit import EncoderUnit

class Encoder(tf.keras.Model):
    
    def __init__(self, num_heads, enc_layers, input_shape, filter_sz):
        super(Encoder, self).__init__()
        
        self.__num_heads  = num_heads
        self.__enc_layers = enc_layers
        self.__filter_sz  = filter_sz
        self.__input_shape= input_shape
        self.__encoder_units  = []
    
        for layer in range(enc_layers):
            encoder_unit = EncoderUnit(num_heads, input_shape[3], filter_sz)
            
            self.__encoder_units.append(encoder_unit)
    

    def call(self, values, training=True):
        
        for layer in range(self.__enc_layers):
            values = self.__encoder_units[layer](values, training)
        
        return values
