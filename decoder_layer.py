import tensorflow as tf

from attention import Attention

class DecoderUnit(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, d_model, filters, filter_size, time_steps):
        super(DecoderUnit, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.atten_weights = []
        self.mask_atten_weights = []
        self.filters = filters + [d_model]
        self.filter_size = filter_size
        self.time_steps = time_steps
        self.attention = Attention()
        self.mask_attention = Attention()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.norm3 = tf.keras.layers.BatchNormalization()
        
        self.layers = []
        for layer in range(len(self.filters)):
            conv_layer = tf.keras.layers.Conv2D(
                self.filters[layer], 
                filter_size, 
                activation='relu',
                padding='same'
            )
            self.layers.append(conv_layer)
        
        # for now, num_heads = 1
        assert(num_heads > 0)
        
        for head in range(num_heads):
            
            self.atten_weights.append([])
            for qkv in range(3):
                weight = tf.keras.layers.Conv2D(d_model, filter_size, padding='same')
                self.atten_weights[head].append(weight)
                
        for head in range(num_heads):
            
            self.mask_atten_weights.append([])
            for qkv in range(3):
                weight = tf.keras.layers.Conv2D(d_model, filter_size, padding='same')
                self.mask_atten_weights[head].append(weight)
                
        
    def call(self, encoder_inputs, decoder_inputs, training=True):
        
        # input shape = (batch_size, time-steps, rows, cols, channels)
        if (decoder_inputs.shape[1] < self.time_steps):
            shape = decoder_inputs.shape
            zeros = tf.zeros([shape[0], self.time_steps - shape[1], shape[2], shape[3], shape[4]])
            decoder_inputs = tf.concat([decoder_inputs, zeros], axis=1)
            
        # masked decoder attention
        query = self.mask_atten_weights[0][0](decoder_inputs)
        keys = self.mask_atten_weights[0][1](decoder_inputs)
        values = self.mask_atten_weights[0][2](decoder_inputs)
        
        mask_values = self.norm1(self.mask_attention(query, keys, values), training=training)
        inputs = tf.concat([encoder_inputs, mask_values], axis=1)

        
        # encoder decoder attention
        query = self.atten_weights[0][0](inputs)
        keys = self.atten_weights[0][1](inputs)
        values = self.atten_weights[0][2](inputs)
        
        
        attentuated = self.norm2(self.attention(query, keys, values), training=training)
        
        # feed forward
        for layer in range(len(self.filters)):
            attentuated = self.layers[layer](attentuated)
            
        return self.norm3(attentuated, training=training)
