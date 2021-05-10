import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/train", label_mode="categorical", batch_size=32,
	image_size=(256, 256), seed=42)
test = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/test", label_mode="categorical", batch_size=32,
	image_size=(256, 256), seed=42)

input = layers.Input(shape=(256, 256, 3))
base_model = tf.keras.applications.ResNet101(input_tensor=input, include_top=True)
model = tf.keras.models.Sequential([
	layers.BatchNormalization(),

	base_model,
	layers.LeakyReLU(),
	layers.BatchNormalization(),
	layers.Flatten(),
	layers.Dense(256),
	layers.LeakyReLU(),
	layers.Dense(128),
	layers.LeakyReLU(),
	layers.Dense(80, activation="softmax", dtype='float32')

])
opt = tf.keras.optimizers.SGD(0.02)
model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
	metrics=['categorical_accuracy'])
history = model.fit(train, validation_data=test, epochs=10)
