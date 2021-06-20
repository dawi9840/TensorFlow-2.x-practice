# Three ways for saving and loading Models example.
import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0 
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0 

# 1. How to save and load model weights.
# 2. Save and load entire model (Serializating model).
#   - Save model weights
#   - Model architecture
#   - Training configuration (model.compile())
#   - Optimizer and states 
model1 = keras.Sequential(
    [
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)

inputs = keras.Input(784)
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)

model3 = MyModel()

# Specify model build from the three API's.
model = model3

# [M1 ~ M3]: Specify model weights load from three ways.
# M1: Load checkpoints weights, neet define model architecture.
# model.load_weights('model_weights/ckpt/')

# M2: Load HDF5 weights, neet define model architecture. 
# However, use model3 to the load ways cannot work. 
# model.load_weights('model_weights/HDF5/model.h5')

# Define model Training configuration, optimizer and states.
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.optimizers.Adam(),
    metrics=["accuracy"],
)

# M3: Load weights and entire model architecture, so don't rebuild and compile model.
# model = keras.models.load_model('model_weights/complete_model/')

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# [S1 ~ S3]: Save weights.
# S1:Checkpoint file.
model.save_weights('model_weights/ckpt/')
print("S1 done!")

# S2: HDF5 file.
model.save_weights('model_weights/HDF5/model.h5')
print("S2 done!")

# S3: Complete model.
model.save('model_weights/complete_model/')
print("S3 done!")