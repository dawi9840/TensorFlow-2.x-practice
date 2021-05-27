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

# CNN model with Functional API.
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, name='1_layer')(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 5, padding='same', name='2_layer')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Conv2D(128, 3, name='3_layer')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64, activation='relu', name='4_layer')(x)

    outputs = layers.Dense(10, name='5_layer')(x) 

    model = keras.Model(inputs=inputs, outputs=outputs, name='Functional_API')
    return model

model = my_model()

# print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
# The predict is about 90%, However the evaluate result is overfitting.
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
