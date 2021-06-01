import tensorflow as tf
import numpy as np
from transformer_video import VideoPrediction
from test import test_model

def load_dataset(path, filename):

    train_data = np.load(path + filename)
    print(train_data.shape, train_data.max(), train_data.min())
    
    return tf.convert_to_tensor(train_data, dtype=tf.float32) / 255.0
        
def main():
    
    inp = load_dataset("../input/ucf101bas20/", 'ucf_reshaped_bas_20.npy')
    X = tf.concat([inp[:300, :5], inp[:300, 10:15]], axis=0)
    Y = tf.concat([inp[:300, 5:10], inp[:300, 15:]], axis=0)
    print(X.shape, Y.shape)

    testX = inp[:, :10]
    testY = inp[:, 10:15]
    print(testX.shape, testY.shape)

    # defines the model
    model = VideoPrediction(
        num_layers=5, d_model=128, num_heads=16, dff=128, filter_size=(3, 3),
        image_shape=X.shape[2:-1], pe_input=10, pe_target=10, out_channel=X.shape[-1]
    )

    model.train(X, Y, testX, testY, 125, 8)
    test_model(model, testX, testY, 8)

if __name__ == "__main__":
    main()
