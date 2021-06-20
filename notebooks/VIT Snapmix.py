import tensorflow as tf
from tensorflow import keras

input_dim = (273, 256, 6)

model = tf.keras.models.Sequential()
base_model = tf.keras.applications.EfficientNetB0(input_shape=input_dim,include_top=False,weights=None)
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))
