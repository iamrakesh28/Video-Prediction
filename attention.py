import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    
    def __init__(self):
        # Assumption : query.shape = values.shape = key.shape
        super(Attention, self).__init__()
        self.__input_shape = None
        self.__softmax = tf.keras.layers.Softmax()
    
    def call(self, query, keys, values):
        # query shape = (batch_size, rows, cols, height)
        
        assert(query.shape == keys.shape == values.shape)
        self.__input_shape = query.shape
        
        no_keys = no_query = self.__input_shape[3]
        weights = []
        
        for q in range(no_query):
            
            activation = []
            for k in range(no_keys):
                dot_prod = query[:, :, :, q] * keys[:, :, :, k]
                dot_prod = tf.reduce_sum(tf.reduce_sum(dot_prod, 1), 1)
                
                activation.append(dot_prod)
                
            activation = tf.transpose(tf.convert_to_tensor(activation))
            activation = self.__softmax(activation)
            
            weight = tf.zeros([self.__input_shape[0], self.__input_shape[1], self.__input_shape[2]])
            for k in range(no_keys):
                score = tf.broadcast_to(
                    tf.reshape(activation[:, k], [self.__input_shape[0], 1, 1]),
                    [self.__input_shape[0], self.__input_shape[1], self.__input_shape[2]])
                
                weight += score * values[:, :, :, k]
            
            weights.append(weight)
        
        weights = tf.transpose(tf.convert_to_tensor(weights), [1, 2, 3, 0])
            
        return weights
