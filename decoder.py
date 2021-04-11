import tensorflow as tf

from decoder_unit import DecoderUnit

class Decoder(tf.keras.Model):
    
    def __init__(
        self, 
        num_heads, 
        dec_layers, 
        d_model,  
        filters, 
        filter_size, 
        enc_steps,
        dec_steps,
        out_channels,
    ):
        super(Decoder, self).__init__()
        
        self.num_heads  = num_heads
        self.dec_layers = dec_layers
        self.filters  = filters
        self.filter_size = filter_size
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.out_channels = out_channels
        self.total_steps = dec_steps + dec_layers * enc_steps
        self.decoder_units  = []
        
        self.final_conv = tf.keras.layers.Conv3D(
            out_channels, 
            (self.total_steps, 1, 1), 
            padding='valid',
            activation='sigmoid',
        )
    
        for layer in range(dec_layers):
            decoder_unit = DecoderUnit(num_heads, d_model, filters, filter_size, dec_steps)
            
            self.decoder_units.append(decoder_unit)
            
    def call(self, enc_values, dec_values, training=True):
        
        for layer in range(self.dec_layers):
            dec_values = self.decoder_units[layer](enc_values, dec_values, training)
        
        return self.final_conv(dec_values)
