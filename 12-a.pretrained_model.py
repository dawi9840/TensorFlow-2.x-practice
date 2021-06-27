# Created a pretrained functional API model.
import os
os.environ['TF_cpp_MIN_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# https://ithelp.ithome.com.tw/articles/10186473
# x_train.reshape(-1:all, width, length, 1:graay)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0 
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0 
print(x_train.shape)
# Functional API: It's a bit more flexible.
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 1, activation='relu')(inputs)
x = layers.Conv2D(64, 1, activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, 1, activation='relu')(x)
x = layers.Conv2D(256, 1, activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='softmax', name='output')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# print(model.summary())

model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=35, verbose=0)
loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=2)

model.save('model_weights/complete_model_0627')

print("Model, accuracy: {:5.2f}%".format(100 * acc))
print("Model saved done!")