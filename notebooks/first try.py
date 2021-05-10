import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from vit_keras import vit, utils

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/train", label_mode="categorical", batch_size=16,
	image_size=(512, 512), seed=42)
test = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/test", label_mode="categorical", batch_size=16,
	image_size=(512, 512), seed=42)
train = train.cache()
test = test.cache()
input = layers.Input(shape=(512, 512, 3))
base_model = tf.keras.applications.ResNet50(include_top=True, weights="imagenet", input_tensor=input)
model = tf.keras.models.Sequential([
	layers.BatchNormalization(),

	base_model,
	layers.LeakyReLU(),
	layers.BatchNormalization(),
	layers.Flatten(),
	layers.Dense(128),
	layers.LeakyReLU(),
	layers.Dense(80, activation="softmax", dtype='float32')

])
opt = tf.keras.optimizers.SGD(0.02)
model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
	metrics=['categorical_accuracy'])
history = model.fit(train, validation_data=test, epochs=5)
