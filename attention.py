import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    
    def __init__(self):
        # Assumption : query.shape = values.shape = key.shape
        super(Attention, self).__init__()
        self.__input_shape = None
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, query, keys, values):
        # query shape = (batch_size, time-steps, rows, cols, channels)
        
        assert(query.shape == keys.shape == values.shape)
        self.__input_shape = query.shape
        
        return values
        dim_querys = dim_keys = self.__input_shape[1]
        weights = []
        
        for q in range(dim_querys):
            
            activation = []
            for k in range(dim_keys):
                dot_prod = query[:, q, :, :, :] * keys[:, k, :, :, :]
                dot_prod = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(dot_prod, 1), 1), 1)
                
                activation.append(dot_prod)
                
            activation = tf.transpose(tf.convert_to_tensor(activation))
            activation = self.softmax(activation / np.sqrt(dim_keys))
            
            weight = tf.zeros(
                [self.__input_shape[0], self.__input_shape[2], self.__input_shape[3], self.__input_shape[4]]
            )
            for k in range(dim_keys):
                score = tf.broadcast_to(
                    tf.reshape(activation[:, k], [self.__input_shape[0], 1, 1, 1]),
                    [self.__input_shape[0], self.__input_shape[2], self.__input_shape[3], self.__input_shape[4]])
                
                weight += score * values[:, k, :, :, :]
            
            weights.append(weight)
        
        weights = tf.transpose(tf.convert_to_tensor(weights), [1, 0, 2, 3, 4])
            
        return weights
