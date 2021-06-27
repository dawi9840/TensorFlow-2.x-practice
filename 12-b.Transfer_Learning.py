# Transfer Learning, Fine Tuning and TensorFlow Hub practice.
import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

# To avoid GPU errors.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#================================================#
#            Pretrained-Model                    #
#================================================#
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0 
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = keras.models.load_model('model_weights/complete_model_0627/')
# print(model.summary())

# Two ways: Froze the pretrained model layers to train.
# way 1:
model.trainable = False 
# way 2:
for layer in model.layers:
    assert layer.trainable == False
    layers.trainable = False

base_inputs = model.layers[0].input
base_output = model.layers[-2].output # Start from flatten layer(-2: last 2 layers). 
final_outputs = layers.Dense(10)(base_output)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
# print(new_model.summary())

new_model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

new_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)



#================================================#
#            Pretrained Keras Model              #
#================================================#
'''
x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

model = keras.applications.InceptionV3(include_top=True)
# print(model.summary())
model.trainable = False 

base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(5)(base_outputs)
new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)

new_model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

new_model.fit(x, y, epochs=15, verbose=2)
'''
import sys 
sys.exit()
#================================================#
#            Pretrained Hub Model                #
#================================================#
x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

# Go https://tfhub.dev/ to choose tfhub.
# tf2-preview/inception_v3/feature_vector: https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4
url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))

# Frozen the pretrained base_model layers.
base_model.trainable = False

# Add new layers to train.
model = keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(5),
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x, y, batch_size=32, epochs=15, verbose=2)
model.evaluate(x, y, batch_size=32, verbose=2)

