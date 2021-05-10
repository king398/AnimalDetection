import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import mixed_precision

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/train", label_mode="categorical", batch_size=16,
	image_size=(128, 128), seed=42)
test = tf.keras.preprocessing.image_dataset_from_directory(
	directory=r"F:/Pycharm_projects/scientificProject/data/test", label_mode="categorical", batch_size=16,
	image_size=(128, 128), seed=42)
train = train.cache()
test = test.cache()
