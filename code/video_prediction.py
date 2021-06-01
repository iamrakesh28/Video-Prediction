import tensorflow as tf
import time

from transformer import Transformer
from utility import create_look_ahead_mask

class VideoPrediction:
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, pe_input, pe_target, out_channel):
        self.transformer = Transformer(num_layers, d_model, num_heads, dff, filter_size,
                                       image_shape, pe_input, pe_target, out_channel)
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer   = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        #self.optimizer   = tf.keras.optimizers.Adam()

    def loss_function(self, real, pred):
        return self.loss_object(real, pred)
    
    def plot_result(self, actual, predict, prev):
        
        for i in range(actual.shape[0]):
            plt.subplot(131), plt.imshow(actual[i]),
            plt.title("Actual_" + str(i + 1))
            plt.subplot(132), plt.imshow(predict[i]),
            plt.title("Predicted_" + str(i + 1))
            plt.subplot(133), plt.imshow(prev[i]),
            plt.title("Previous_" + str(i + 1))
            plt.show()
        
    def train_step(self, inp, tar, idx):
        
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
        loss = 0

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, True, look_ahead_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        return loss

    def train(self, inp, tar, epochs, batch_size):

        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            total_batch = inp.shape[0] // batch_size
            
            for batch in range(total_batch):
                index = batch * batch_size
                enc_inp = inp[index:index + batch_size, :, :, :]
                dec_inp = tar[index:index + batch_size, :, :, :]
                
                batch_loss = self.train_step(enc_inp, dec_inp, idx)
                total_loss += batch_loss
                
                # saving (checkpoint) the model every 25 epochs
                # if epoch % 50 == 0:
                # self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                # val_loss = self.evaluate(valX, valY)
                # print('Epoch {} Evaluation Loss {:.4f}'.format(epoch + 1, val_loss))
                # if epoch % 50 == 0:
                # test_model(self, X, Y)
                # if (time.time() - init_time) / 3600.0 > 8:
                #    break

            total_batch += 1
            if epoch % 5 == 0:
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / total_batch))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def predict(self, inp, tar_seq_len):

        inp = tf.expand_dims(inp, 0)
        image_size = inp.shape[2:]
        
        look_ahead_mask = create_look_ahead_mask(20)
        
        predictions = []
        output = inp[:, 10:, :, :, :]
        encoder_input = inp[:, :10, :, :, :]
        
        for t in range(tar_seq_len):
            prediction, _ = self.transformer(
                encoder_input, output, False, look_ahead_mask
            )
                
            predict = prediction[:, -1:, :, :, :]
            
            output = tf.concat([output, predict], axis=1)
            predictions.append(
                predict.numpy().reshape(
                    image_size
                )
            )
            
        return np.array(predictions)
