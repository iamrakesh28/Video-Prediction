import numpy as np
import matplotlib.pyplot as plt

d_model = 40 * 40
base = 10000.0

def f(x, y):
    return np.sin(x) * np.cos(y)

def plot_sin_2D(rows, cols, pos):

    freq = []
    for i in range(rows):
        freq.append([])

        for j in range(cols):
            freq[i].append(f(pos / (base ** (2 * np.pi * i / d_model)),
                             pos / (base ** (2 * np.pi * j / d_model))))

    return np.array(freq)

def plot_sin_1D(position):
    
    idx = np.arange(1, d_model + 1, 1.0)
    sinx = np.sin(position / (base ** (idx[1::2] / d_model)))
    cosx = np.cos(position / (base ** (idx[0::2] / d_model)))
    
    embedding = np.zeros((d_model, ))
    embedding[1::2] = sinx
    embedding[0::2] = cosx

    embedding = embedding.reshape((40, 40))

    return embedding

    

img = plot_sin_2D(100, 100, 5.0)
plt.imshow(img)
plt.show()

'''
plt.subplot(2, 4, 1), plt.imshow(plot_sin_1D(1.0))
plt.subplot(2, 4, 2), plt.imshow(plot_sin_1D(2.0))
plt.subplot(2, 4, 3), plt.imshow(plot_sin_1D(3.0))
plt.subplot(2, 4, 4), plt.imshow(plot_sin_1D(4.0))
plt.subplot(2, 4, 5), plt.imshow(plot_sin_1D(5.0))
plt.subplot(2, 4, 6), plt.imshow(plot_sin_1D(6.0))
plt.subplot(2, 4, 7), plt.imshow(plot_sin_1D(7.0))
plt.subplot(2, 4, 8), plt.imshow(plot_sin_1D(8.0))
plt.show()
'''
