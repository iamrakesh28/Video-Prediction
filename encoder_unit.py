import tensorflow as tf

from attention import Attention

class EncoderUnit(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, num_images, filter_size):
        super(EncoderUnit, self).__init__()
        self.__num_heads = num_heads
        self.__atten_conv2d = []
        self.__num_images = num_images
        self.__filter_size = filter_size
        self.__attention = Attention()
        self.__norm1 = tf.keras.layers.BatchNormalization()
        self.__norm2 = tf.keras.layers.BatchNormalization()
        # filter number may be changed
        self.__layer1 = tf.keras.layers.Conv2D(
            2 * num_images, 
            filter_size, 
            activation='relu',
            padding='same'
        )
        
        self.__layer2 = tf.keras.layers.Conv2D(
            num_images, 
            filter_size, 
            activation='relu',
            padding='same'
        )
        
        # for now, num_heads = 1
        assert(num_heads > 0)
        
        for head in range(num_heads):
            
            self.__atten_conv2d.append([])
            for qkv in range(3):
                conv2d = tf.keras.layers.Conv2D(num_images, filter_size, padding='same')
                self.__atten_conv2d[head].append(conv2d)
                
        
    def call(self, inputs, training=True):
        
        # self-attention
        query = self.__atten_conv2d[0][0](inputs)
        keys = self.__atten_conv2d[0][1](inputs)
        values = self.__atten_conv2d[0][2](inputs)
        
        
        attentuated = self.__norm1(self.__attention(query, keys, values), training=training)
        
        # feed forward
        return self.__norm2(self.__layer2(self.__layer1(attentuated)), training=training)
