import tensorflow as tf

from decoder_unit import DecoderUnit

class Decoder(tf.keras.Model):
    
    def __init__(self, num_heads, dec_layers, encoder_num_imgs, decoder_num_imgs, filter_sz):
        super(Decoder, self).__init__()
        
        self.__num_heads  = num_heads
        self.__dec_layers = dec_layers
        self.__filter_sz  = filter_sz
        # self.__input_shape= input_shape
        self.__decoder_units  = []
        self.__final_conv = tf.keras.layers.Conv2D(
            1, 
            filter_sz, 
            padding='same',
            activation='relu'
        )
    
        for layer in range(dec_layers):
            decoder_unit = DecoderUnit(num_heads, encoder_num_imgs, decoder_num_imgs, filter_sz)
            
            self.__decoder_units.append(decoder_unit)
            
    def call(self, enc_values, dec_values, training=True):
        
        for layer in range(self.__dec_layers):
            dec_values = self.__decoder_units[layer](enc_values, dec_values, training)
        
        return self.__final_conv(dec_values)
