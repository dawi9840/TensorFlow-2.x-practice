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
print("x_train: ", x_train.shape) # (60000, 28, 28)
print("y_train: ", y_train.shape) # (60000,)

# Preprocessing: Flatten the data. -1: keep 6000; 28*28: flatten. 
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0 
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0 

# Sequential API: It's very convenient but not very flexible.
# Sequential method1:
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10), 
    ]
)

# Sequential method2:
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu', name='first_layer'))
model.add(layers.Dense(256, activation='relu', name='test_layer'))
model.add(layers.Dense(10, name='output'))

print(model.summary())

# Debug Skills: Override the model and specify the output layer to debug.
# Debug Method1:
model = keras.Model(inputs=model.inputs,
                    # Will get the specify output layer.
                    outputs=[model.layers[-2].output])
# Debug Method2:
model = keras.Model(inputs=model.inputs,
                    outputs=[model.get_layer('test_layer').output])
 
feature = model.predict(x_train)
print("Specify layer shape: ", feature.shape)

# Debug Method3:
model = keras.Model(inputs=model.inputs,
                    outputs=[layer.output for layer in model.layers])

features = model.predict(x_train)
print("Show all features shape: ")
for feature in features:
    print( feature.shape)

# Interrupt the following code.
import sys
sys.exit()

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
