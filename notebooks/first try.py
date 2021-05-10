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
	directory=r"F:/Pycharm_projects/scientificProject/data/train", label_mode="categorical", batch_size=16,
	image_size=(256, 256), seed=42)
test = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/test", label_mode="categorical", batch_size=16,
	image_size=(256, 256), seed=42)
train = train.cache()
test = test.cache()
input = layers.Input(shape=(256, 256, 3))
base_model = tf.keras.applications.EfficientNetB1(include_top=True, weights="imagenet", input_tensor=input)
model = tf.keras.models.Sequential([
	base_model,
	layers.Flatten(),
	layers.Dense(80, activation="softmax")

])
opt = tf.keras.optimizers.SGD(0.03)
model.compile(
	optimizer=opt,
	loss="categorical_crossentropy",
	metrics=['categorical_accuracy'])
history = model.fit(train, validation_data=test, epochs=5)
