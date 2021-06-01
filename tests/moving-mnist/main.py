import tensorflow as tf
import numpy as np
from transformer_video import VideoPrediction
from test import test_model

def load_dataset(path, filename):
    
    train_data = np.load(path + filename)
    train_data[[1005, 9000]] = train_data[[9000, 1005]]
    
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1
    # train_data = train_data / 255.0
    print(train_data.shape)
    
    X = train_data[:, :10, :, :, :]
    Y = train_data[:, 10:21, :, :, :]

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    
    X = tf.concat([X[:1000], X[9000:]], axis=0)
    Y = tf.concat([Y[:1000], Y[9000:]], axis=0)
    
    return (X, Y)
        
def main():
    
    X, Y = load_dataset("../input/mnistreshape/", 'mnist-reshape.npy')

    # defines the model
    model = VideoPrediction(
        num_layers=5, d_model=64, num_heads=16, dff=128, filter_size=(3, 3),
        image_shape=X.shape[2:-1], pe_input=10, pe_target=10, out_channel=X.shape[-1]
    )
    
    # training on first 1000 samples
    # samples from 1000 - 1199 are used as test set
    model.train(X[:1000, :5], X[:1000, 5:], X, Y, 100, 8)

    test_model(model, X, Y, 8)

if __name__ == "__main__":
    main()
