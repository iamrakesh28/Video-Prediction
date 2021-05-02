import tensorflow as tf

INFINITY = 1e9

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights

    Args:
    q : query shape = (..., seq_len, rows, cols, depth)
    k : keys shape = (..., seq_len, rows, cols, depth)
    v : values shape = (..., seq_len, rows, cols, depth)
    
    Returns:
    outputs, attention_weights
    """

    assert (q.shape == k.shape == v.shape)
    seq_len = q.shape[-4]
    dim_k = tf.cast(k.shape[-1], tf.float32)

    attention_weights = []
    outputs = []
    for seq in range(seq_len):

        query = None
        if len(q.shape) == 6:
            query = tf.expand_dims(q[:, :, seq, :, :, :], axis=2)
        else: # len(q.shape) = 5
            query = tf.expand_dims(q[:, seq, :, :, :], axis=1)
            
        scaled_dot_product = tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(query * k, axis=-1),
                axis=-1
            ),
            axis=-1
        ) / tf.math.sqrt(dim_k)

        if mask is not None:
            scaled_dot_product += (mask * -INFINITY)
                                   
        attention_weight = tf.nn.softmax(scaled_dot_product, axis=-1)
        output = tf.expand_dims(
            tf.expand_dims(
                tf.expand_dims(attention_weight, axis=-1),
                axis=-1
            ),
            axis=-1
        ) * v

        output = tf.reduce_sum(output, axis=-4)

        attention_weights.append(attention_weight)
        outputs.append(output)

    attention_weights = tf.convert_to_tensor(attention_weights)
    outputs = tf.convert_to_tensor(outputs)

    trans_weight = None
    trans_out = None
    if len(q.shape) == 6:
        trans_out = [1, 2, 0, 3, 4, 5]
        trans_weight = [1, 2, 0, 3]
    else: # len(q.shape) = 5
        trans_out = [1, 0, 2, 3, 4]
        trans_weight = [1, 0, 2]

    attention_weights = tf.transpose(attention_weights, trans_weight)
    outputs = tf.transpose(outputs, trans_out)

    return outputs, attention_weights
    
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


x = tf.random.normal([2, 2, 3, 4, 4, 3])
a, b = scaled_dot_product_attention(x, x, x, None)
print(a.shape, b.shape, x.shape)
