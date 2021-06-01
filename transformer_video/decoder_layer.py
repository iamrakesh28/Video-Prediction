import tensorflow as tf

from .utility import feed_forward_network
from .multi_head_attention import MultiHeadAttention

class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, dff, filter_size):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads, filter_size)
        self.mha2 = MultiHeadAttention(d_model, num_heads, filter_size)

        self.ffn = feed_forward_network(dff, d_model, filter_size)
        
        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        
        # No dropouts for now
                
        
    def call(self, x, enc_output, training, look_ahead_mask):

        # enc_output.shape = (batch_size, input_seq_len, rows, cols, d_model)
        # x.shape = (batch_size, input_seq_len, rows, cols, d_model)
        
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # (batch_size, target_seq_len, rows, cols, d_model)
        
        out1 = self.layernorm1(x + attn1, training=training)
        # (batch_size, target_seq_len, rows, cols, d_model)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, None)
        # (batch_size, target_seq_len, rows, cols, d_model)
        
        out2 = self.layernorm2(out1 + attn2, training=training)
        # (batch_size, target_seq_len, rows, cols, d_model)

        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, rows, cols, d_model)
        out3 = self.layernorm3(out2 + ffn_output, training=training)
        
        return out3, attn_weights_block1, attn_weights_block2

