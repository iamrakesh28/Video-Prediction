import tensorflow as tf

from multi_head_attention import MultiHeadAttention

class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, dff, filter_size):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, filter_size)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dff, filter_size, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(d_model, filter_size, padding='same')
        ])
        
        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        
        # No dropouts for now
                
        
    def call(self, x, training, mask):
        
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, rows, cols, d_model)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, rows, cols, d_model)
        
        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, rows, cols, d_model)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

