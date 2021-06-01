import tensorflow as tf
import numpy as np
from transformer_video import VideoPrediction

def generate_movies(n_samples=1200, n_frames=20):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies[shifted_movies >= 1] = 1
    return shifted_movies
    
    
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
    
    shifted_movies = tf.convert_to_tensor(generate_movies(n_samples=1200), dtype=tf.float32)
    print(shifted_movies.shape)

    X = shifted_movies[:, :10, :, :, :]
    Y = shifted_movies[:, 10:, :, :, :]
    
    # defines the model
    model = VideoPrediction(
        num_layers=3, d_model=64, num_heads=16, dff=128,
        filter_size=(3, 3), image_shape=(40, 40), pe_input=10,
        pe_target=20, out_channel=1, loss_function='bin_cross'
    )
    
    # training on first 1000 samples
    # samples from 1000 - 1199 are used as test set
    model.train(X[:1000, :5], X[:1000, 5:], None, None, 1, 8)

    x1 = tf.concat((X[1036], Y[1036]), axis=0)
    x2 = tf.concat((X[1017], Y[1017]), axis=0)
    y1 = model.predict(x1[:10], 10)
    y2 = model.predict(x2[:10], 10)
    x1 = x1.numpy().reshape(20, 40, 40)
    x2 = x2.numpy().reshape(20, 40, 40)
    plot_result(x1[:10], x1[10:], y1.reshape(10, 40, 40))
    plot_result(x2[:10], x2[10:], y2.reshape(10, 40, 40))

    x1 = tf.concat((X[1026], Y[1026]), axis=0)
    x2 = tf.concat((X[1027], Y[1027]), axis=0)
    y1 = model.predict(x1[:10], 10)
    y2 = model.predict(x2[:10], 10)
    x1 = x1.numpy().reshape(20, 40, 40)
    x2 = x2.numpy().reshape(20, 40, 40)
    plot_result(x1[:10], x1[10:], y1.reshape(10, 40, 40))
    plot_result(x2[:10], x2[10:], y2.reshape(10, 40, 40))
    

if __name__ == "__main__":
    main()
