import tensorflow as tf

INFINITY = 1e9

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the attention weights

    Args:
    q : query shape = (batch_sz, heads, seq_len_q, rows, cols, depth)
    k : keys shape = (batch_sz, heads, seq_len_k, rows, cols, depth)
    v : values shape = (batch_sz, heads, seq_len_v, rows, cols, depth)
    mask : shape = (batch_sz, heads, seq_len, rows, cols, depth)
    
    Returns:
    outputs, attention_weights
    """

    seq_len = q.shape[2]
    dim_k = tf.cast(k.shape[-1], tf.float32)

    attention_weights = []
    outputs = []
    for seq in range(seq_len):

        query = tf.expand_dims(q[:, :, seq, :, :, :], axis=2)
            
        scaled_dot_product = tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(query * k, axis=-1),
                axis=-1
            ),
            axis=-1
        ) / tf.math.sqrt(dim_k)

        if mask is not None:
            mask = tf.concat(
                (tf.zeros(mask[:, :, :seq + 1].shape), mask[:, :, :seq_len - seq - 1]),
                axis=-1
            )
            scaled_dot_product += (mask * -INFINITY)
                                   
        attention_weight = tf.nn.softmax(scaled_dot_product, axis=-1)
        output = tf.expand_dims(
            tf.expand_dims(
                tf.expand_dims(attention_weight, axis=-1),
                axis=-1
            ),
            axis=-1
        ) * v

        output = tf.reduce_sum(output, axis=2)

        attention_weights.append(attention_weight)
        outputs.append(output)

    attention_weights = tf.convert_to_tensor(attention_weights)
    outputs = tf.convert_to_tensor(outputs)

    attention_weights = tf.transpose(attention_weights, perm=[1, 2, 0, 3])
    outputs = tf.transpose(outputs, perm=[1, 2, 0, 3, 4, 5])

    return outputs, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, filter_size):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.filter_size = filter_size

        assert (d_model % num_heads == 0)

        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Conv2D(d_model, filter_size, padding='same', activation='relu')
        self.wk = tf.keras.layers.Conv2D(d_model, filter_size, padding='same', activation='relu')
        self.wv = tf.keras.layers.Conv2D(d_model, filter_size, padding='same', activation='relu')

        self.final_weight = tf.keras.layers.Conv2D(d_model, filter_size, padding='same')

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, rows, cols, depth)
        
        Args:
        x : shape = (batch_size, seq_len, rows, cols, depth)
        """
        x = tf.reshape(x, x.shape[:4] + (self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 4, 1, 2, 3, 5])
        
    def call(self, q, k, v, mask=None):
        
        shape_q = q.shape
        q = self.wq(q) # (batch_size, seq_len_q, rows, cols, depth)
        k = self.wk(k) # (batch_size, seq_len_k, rows, cols, depth)
        v = self.wv(v) # (batch_size, seq_len_v, rows, cols, depth)

        q = self.split_heads(q) # (batch_size, num_heads, seq_len_q, rows, cols, depth)
        k = self.split_heads(k) # (batch_size, num_heads, seq_len_k, rows, cols, depth)
        v = self.split_heads(v) # (batch_size, num_heads, seq_len_v, rows, cols, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, rows, cols, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 3, 4, 1, 5])
        # (batch_size, seq_len_q, rows, cols, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, shape_q)

        output = self.final_weight(concat_attention)

        return output, attention_weights
