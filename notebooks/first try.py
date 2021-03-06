import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from cutmix_keras import CutMixImageDataGenerator  # Import CutMix
from tensorflow.keras.models import load_model
from efficientnet.keras import EfficientNetB0


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.keras.regularizers.l2(l2=0.01)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1. / 255, horizontal_flip=True, dtype=tf.float32)

train1 = train_datagen.flow_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/train", class_mode="categorical", batch_size=16,
	target_size=(384, 384), seed=42, shuffle=True)
train2 = train_datagen.flow_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/train", class_mode="categorical", batch_size=16,
	target_size=(384, 384), seed=42, shuffle=True)
test = train_datagen.flow_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/valid", class_mode="categorical", batch_size=16,
	target_size=(384, 384), seed=42, shuffle=True)
train = CutMixImageDataGenerator(
	generator1=train1,
	generator2=train2,
	img_size=384,
	batch_size=16,
)

input = layers.Input(shape=(384, 384, 3))
base_model = load_model(r"F:\Pycharm_projects\scientificProject\models\efficientnet-b1_noisy-student_notop.h5",compile=False)
model = tf.keras.models.Sequential([
	layers.BatchNormalization(),

	base_model,
	layers.LeakyReLU(),
	layers.BatchNormalization(),
	layers.Flatten(),
	layers.Dense(384),
	layers.LeakyReLU(),
	layers.Dense(128),
	layers.LeakyReLU(),
	layers.Dense(80, activation="softmax", dtype='float32')

])
opt = tf.keras.optimizers.SGD(0.02)
model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
	metrics=['categorical_accuracy'])
history = model.fit(train, validation_data=test, epochs=10, steps_per_epoch=1410.375)
