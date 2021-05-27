import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0 

# CNN model wuth Sequential. 
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', name='1_layer'),
        layers.MaxPooling2D(pool_size=(2,2), name='2_layer'),
        layers.Conv2D(64, 3, padding='valid', activation='relu', name='3_layer'),
        layers.MaxPooling2D(name='4_layer'),
        layers.Conv2D(128, 3, activation='relu',name='5_layer'),
        layers.Flatten(name='6_layer'),
        layers.Dense(64, activation='relu',name='7_layer'),
        layers.Dense(10, name='8_layer'), 
    ]
)

# print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
