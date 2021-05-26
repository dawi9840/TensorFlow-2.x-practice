import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load datasets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing: Flatten the data. 
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0 
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0 

# Functional API: It's a bit more flexible.
inputs = keras.Input(shape=(784), name='input')
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax', name='output')(x)

# Method1:
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

# Interrupt the following code.
import sys
sys.exit()

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = tf.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
