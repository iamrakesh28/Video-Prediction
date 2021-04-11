import tensorflow as tf
import numpy as np
from encoder_decoder import EncoderDecoder
from test import test_model

def load_dataset(path, filename):
    train_data = np.load(path + filename)
    train_data = train_data.swapaxes(0, 1)[:100]
    # train_data[[1005, 9000]] = train_data[[9000, 1005]]

    # patch size 2 x 2
    # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 16, 16, 16)
    #train_data = reshape_patch(train_data, (2, 2))
    #plt.imshow(restore_patch(train_data[0], (2, 2))[0])
    #plt.show()
    
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1
    #train_data = train_data / 255.0
    print(train_data.shape)
    
    train_data = np.expand_dims(train_data, axis=4)
    X = train_data[:, :10, :, :, :]
    Y = train_data[:, 10:21, :, :, :]
    

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    
    return (X, Y)
    
def plot_result(input_, actual, predict):
    
    for i in range(input_.shape[0]):
        plt.imshow(input_[i])
        plt.title("Actual_" + str(i + 1))
        plt.show()
        
    for i in range(actual.shape[0]):
        plt.subplot(121), plt.imshow(actual[i]),
        plt.title("Actual_" + str(i + 1 + input_.shape[0]))
        plt.subplot(122), plt.imshow(predict[i]),
        plt.title("Predicted_" + str(i + 1 + input_.shape[0]))
        plt.show()
        
def main():
    
    tf.debugging.set_log_device_placement(False)
    X, Y = load_dataset("../input/mnist-cs/", 'mnist_test_seq.npy')
    
    model = EncoderDecoder(
        1,
        1,
        10,
        [16, 16], 
        (3, 3),
        4,
        X.shape,
        Y.shape[1],
        Y.shape[4],
        './training_checkpoints'
    )
    
    #model.restore()
    model.train(X[:100], Y[:100], 50, X[100:100], Y[100:100], X, Y)

    test_model(model, X, Y)
    

if __name__ == "__main__":
    main()
