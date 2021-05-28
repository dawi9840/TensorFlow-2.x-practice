# Example for using RNNs, GRUs, LSTMs and Bidirectionality.
import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0 

def SimpleRNN_example():
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, 28))) # We don't specify the times, so the dimension to set None with 28 pixel.
    model.add(
        layers.SimpleRNN(512, return_sequences=True, activation='relu')
    )
    model.add(layers.SimpleRNN(512, activation='relu'))
    model.add(layers.Dense(10))
    return model

def GRU_example():
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, 28)))
    model.add(
        layers.GRU(256, return_sequences=True, activation='tanh')
    )
    model.add(layers.GRU(256, activation='tanh'))
    model.add(layers.Dense(10))
    return model

def LSTM_example():
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, 28)))
    model.add(
        layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, activation='tanh')
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(256, activation='tanh')
        )
    )
    model.add(layers.Dense(10))
    return model

model = LSTM_example()

# print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Because not have softmax, the logits set true.
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

