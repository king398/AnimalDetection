import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from cutmix_keras import CutMixImageDataGenerator  # Import CutMix

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1. / 255,
)

train = train_datagen.flow_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/train", class_mode="categorical", batch_size=32,
	target_size=(256, 256), seed=42, shuffle=True)
test = train_datagen.flow_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/test", class_mode="categorical", batch_size=32,
	target_size=(256, 256), seed=42, shuffle=True)


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
