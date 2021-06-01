import tensorflow as tf
import numpy as np

BASE = 10000

def feed_forward_network(dff, d_model, filter_size):

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(dff, filter_size, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(d_model, filter_size, padding='same')
    ])

def get_angles(pos, i, base_dim, d_model):
    angle_rates = 1 / np.power(BASE, (2 * (i // 2)) / np.float32(d_model))
    angle_rates = np.broadcast_to(
        np.reshape(angle_rates, (1, 1, 1, -1)),
        (1, base_dim[0], base_dim[1], d_model)
    )
    return pos * angle_rates
    
def positional_encoding(position, base_dim, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis, np.newaxis, np.newaxis],
        np.arange(d_model)[np.newaxis, np.newaxis, np.newaxis, :],
        base_dim,
        d_model
    )

    # apply sin to even indices; 2i
    angle_rads[:, :, :, 0::2] = np.sin(angle_rads[:, :, :, 0::2])

    # apply cos to odd indices; 2i + 1
    angle_rads[:, :, :, 1::2] = np.cos(angle_rads[:, :, :, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_look_ahead_mask(seq_len):

    return tf.ones((1, 1, seq_len), dtype=tf.float32)
