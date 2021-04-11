import tensorflow as tf

from attention import Attention

class EncoderUnit(tf.keras.layers.Layer):
    
    '''
        @num_heads number of attention heads
        @d_model channel dimension (equivalent to d_model in 'Attention is all you need')
        @filter filter numbers for feed forward layers [f1, f2, .., fn]
    '''
    def __init__(self, num_heads, d_model, filters, filter_size):
        super(EncoderUnit, self).__init__()
        self.num_heads = num_heads
        self.atten_weights = []
        self.d_model = d_model
        self.filter_size = filter_size
        self.filters = filters + [d_model]
        self.attention = Attention()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
        
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
                
        
    def call(self, inputs, training=True):
        
        # self-attention
        query = self.atten_weights[0][0](inputs)
        keys = self.atten_weights[0][1](inputs)
        values = self.atten_weights[0][2](inputs)
        
        
        attentuated = self.norm1(self.attention(query, keys, values), training=training)
        
        # feed forward
        for layer in range(len(self.filters)):
            attentuated = self.layers[layer](attentuated)
            
        return self.norm2(attentuated, training=training)
