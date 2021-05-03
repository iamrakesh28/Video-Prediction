import tensorflow as tf

def feed_forward_network(dff, d_model, filter_size):

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(dff, filter_size, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(d_model, filter_size, padding='same')
    ])
