import tensorflow as tf

from attention import Attention

class DecoderUnit(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, encoder_num_imgs, decoder_num_imgs, filter_size):
        super(DecoderUnit, self).__init__()
        self.__num_decoder_atten = decoder_num_imgs
        self.__num_heads = num_heads
        self.__atten_conv2d = []
        self.__mask_atten_conv2d = []
        self.__encoder_num_imgs = encoder_num_imgs
        self.__decoder_num_imgs = decoder_num_imgs
        self.__filter_size = filter_size
        self.__attention = Attention()
        self.__mask_attention = Attention()
        self.__norm1 = tf.keras.layers.BatchNormalization()
        self.__norm2 = tf.keras.layers.BatchNormalization()
        self.__norm3 = tf.keras.layers.BatchNormalization()
        
        # filter number may be changed
        self.__layer1 = tf.keras.layers.Conv2D(
            decoder_num_imgs + encoder_num_imgs, 
            filter_size,
            activation='relu',
            padding='same'
        )
        self.__layer2 = tf.keras.layers.Conv2D(
            decoder_num_imgs, 
            filter_size, 
            padding='same',
            activation='relu'
        )
        
        # for now, num_heads = 1
        assert(num_heads > 0)
        
        for head in range(num_heads):
            
            self.__atten_conv2d.append([])
            for qkv in range(3):
                conv2d = tf.keras.layers.Conv2D(
                    decoder_num_imgs + encoder_num_imgs, 
                    filter_size, 
                    padding='same'
                )
                self.__atten_conv2d[head].append(conv2d)
                
        for head in range(num_heads):
            
            self.__mask_atten_conv2d.append([])
            for qkv in range(3):
                conv2d = tf.keras.layers.Conv2D(decoder_num_imgs, filter_size, padding='same')
                self.__mask_atten_conv2d[head].append(conv2d)
                
        
    def call(self, encoder_inputs, decoder_inputs, training=True):
        
        # input shape = (batch_size, rows, cols, height)
        assert(decoder_inputs.shape[3] <= self.__num_decoder_atten)
        if (decoder_inputs.shape[3] < self.__num_decoder_atten):
            shape = decoder_inputs.shape
            zeros = tf.zeros([shape[0], shape[1], shape[2], self.__num_decoder_atten - shape[3]])
            decoder_inputs = tf.concat([decoder_inputs, zeros], axis=3)
            
        # masked decoder attention
        query = self.__mask_atten_conv2d[0][0](decoder_inputs)
        keys = self.__mask_atten_conv2d[0][1](decoder_inputs)
        values = self.__mask_atten_conv2d[0][2](decoder_inputs)
        
        mask_values = self.__norm1(self.__mask_attention(query, keys, values), training=training)
        inputs = tf.concat([encoder_inputs, mask_values], axis=3)

        
        # encoder decoder attention
        query = self.__atten_conv2d[0][0](inputs)
        keys = self.__atten_conv2d[0][1](inputs)
        values = self.__atten_conv2d[0][2](inputs)
        
        
        attentuated = self.__norm2(self.__attention(query, keys, values), training=training)
        
        # feed forward
        return self.__norm3(self.__layer2(self.__layer1(attentuated)), training=training)
