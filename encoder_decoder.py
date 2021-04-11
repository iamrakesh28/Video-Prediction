import tensorflow as tf
import numpy as np
import time
import os

from encoder import Encoder
from decoder import Decoder

# Builds an encoder-decoder
class EncoderDecoder:
    def __init__(
        self, 
        num_heads,
        num_layers,
        d_model,
        filters, 
        filter_size,
        batch_sz,
        input_shape,
        out_steps,
        out_channels,
        checkpoint_dir,
    ):
        self.num_layers  = num_layers
        self.batch_sz    = batch_sz
        self.__input_shape = input_shape
        self.encoder     = Encoder(num_heads, num_layers, d_model, filters, filter_size)
        self.decoder     = Decoder(
            num_heads, num_layers, d_model, filters, filter_size, input_shape[1] - 1, out_steps, out_channels
        )
        self.optimizer   = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        self.out_channels = out_channels
        # self.optimizer   = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.checkpoint_dir = checkpoint_dir
        # self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # self.checkpoint = tf.train.Checkpoint(
        #    optimizer=self.optimizer, 
        #    encoder=self.encoder, 
        #    decoder=self.decoder
        # )
        
        # Binary crossentropy
        # T * logP + (1 - T) * log(1 - P)
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        #self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        #self.loss_object = tf.keras.losses.MSE
        # self.loss_object = tf.keras.losses.CategoricalCrossentropy(
        #    reduction=tf.keras.losses.Reduction.SUM
        #)
        
    def __loss_function(self, real_frame, pred_frame):
        return self.loss_object(real_frame, pred_frame)
        
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    def __train_step(self, input_, target):
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        with tf.GradientTape() as tape:

            enc_values = self.encoder(input_[:, :start_pred, :, :, :], True)
            dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
            # Teacher forcing
            for t in range(0, target.shape[1]):
                prediction = self.decoder(enc_values, dec_input, True)
                actual = tf.expand_dims(target[:, t, :, :, :], 1)
                
                batch_loss += self.__loss_function(actual, prediction)
                
                # using teacher forcing
                dec_input = tf.concat([dec_input, actual], 1)
        

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return (batch_loss / int(target.shape[1]))
    
    # inputX - > (total, time_steps, rows, cols, channels)
    # targetY -> (total, time_steps, rows, cols, channels)
    def train(self, inputX, targetY, epochs, valX, valY, X, Y):
        init_time = time.time()
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            total_batch = inputX.shape[0] // self.batch_sz
            #print(total_batch)
            
            for batch in range(total_batch):
                index = batch * self.batch_sz
                input_ = inputX[index:index + self.batch_sz, :, :, :]
                target = targetY[index:index + self.batch_sz, :, :, :]
                
                # print(input_.shape, target.shape)
                
                batch_loss = self.__train_step(input_, target)
                total_loss += batch_loss
                
            # saving (checkpoint) the model every 25 epochs
            if epoch % 10 == 0:
                # self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                val_loss = self.evaluate(valX, valY)
                print('Epoch {} Evaluation Loss {:.4f}'.format(epoch + 1, val_loss))
                # if epoch % 50 == 0:
                test_model(self, X, Y)
                if (time.time() - init_time) / 3600.0 > 8:
                    break

            total_batch += 1
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / total_batch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    # valid  -> validation
    def __eval_step(self, input_, target, valid):
        
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        enc_values = self.encoder(input_[:, :start_pred, :, :, :], True)
        dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
        # Teacher forcing
        for t in range(0, target.shape[3]):
            prediction = self.decoder(enc_values, dec_input, True)
            actual = tf.expand_dims(target[:, t, :, :, :], 1)
            batch_loss += self.__loss_function(actual, prediction)

            # if evaluating on validation set
            if valid:
                # using teacher forcing
                dec_input = tf.concat([dec_input, actual], 1)
            else:
                # evaluating on testing set
                dec_input = tf.concat([dec_input, prediction], 1)
        
        return (batch_loss / int(target.shape[1]))

    # input_ -> (time_steps, rows, cols, channels)
    def predict(self, input_, output_seq):
        input_ = tf.expand_dims(input_, 0)
        start_pred = input_.shape[1] - 1
        enc_values = self.encoder(input_[:, :start_pred, :, :, :], False)
        dec_input = tf.expand_dims(input_[:, -1, :, :, :], 1)
        
        predictions = []
        
        for t in range(output_seq):
            prediction = self.decoder(enc_values, dec_input, False)
            dec_input = tf.concat([dec_input, prediction], 1)
            predictions.append(
                prediction.numpy().reshape(
                    (self.__input_shape[2], self.__input_shape[3], self.out_channels)
                )
            )
            
        return np.array(predictions)
    
    def evaluate(self, inputX, outputY, valid=True):
        
        total_loss = 0
        total_batch = inputX.shape[0] // self.batch_sz
        
        for batch in range(total_batch):
            index = batch * self.batch_sz
            input_ = inputX[index:index + self.batch_sz, :, :, :, :]
            target = outputY[index:index + self.batch_sz, :, :, :, :]
                
            batch_loss = self.__eval_step(input_, target, valid)
            total_loss += batch_loss
    
        total_batch += 1
        return total_loss / total_batch
