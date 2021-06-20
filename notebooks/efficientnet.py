import tensorflow as tf
import numpy as np
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from cutmix_keras import CutMixImageDataGenerator  # Import CutMix
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
import tensorflow_addons as tfa

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.keras.regularizers.l2(l2=0.01)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1. / 255, dtype=tf.float32)

train1 = train_datagen.flow_from_directory(
	directory=r"/content/train", class_mode="categorical", batch_size=12,
	target_size=(384, 384), seed=42, shuffle=True)
train2 = train_datagen.flow_from_directory(
	directory=r"/content/train", class_mode="categorical", batch_size=12,
	target_size=(384, 384), seed=42, shuffle=True)
test = train_datagen.flow_from_directory(
	directory=r"/content/test", class_mode="categorical", batch_size=12,
	target_size=(384, 384), seed=42, shuffle=True)
train = CutMixImageDataGenerator(
	generator1=train1,
	generator2=train2,
	img_size=384,
	batch_size=12,
)

input = layers.Input(shape=(384, 384, 3))
base_model = vit.vit_b16(
	image_size=384,
	activation="softmax",
	pretrained=True,
	include_top=True,
	pretrained_top=True
)
classes = utils.get_imagenet_classes()
url = r'/content/train/Lion/0b42c367365139bb.jpg'
image = utils.read(url, 384)
attention_map = visualize.attention_map(model=base_model, image=image)
print('Prediction:', classes[
	base_model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
      )
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(image)
_ = ax2.imshow(attention_map)
plt.show()

model = tf.keras.models.Sequential([
	layers.BatchNormalization(),

	base_model,
	layers.LeakyReLU(),
	layers.BatchNormalization(renorm=True),
	layers.Flatten(),
	layers.Dense(256),
	layers.LeakyReLU(),
	layers.Dense(128),
	layers.LeakyReLU(),
	layers.Dense(80, activation="softmax", dtype='float32')

])
checkpoint_filepath = r"/content/temp"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)

opt = tf.keras.optimizers.SGD(0.03)
model.compile(
	optimizer=opt,
	loss=tf.losses.CategoricalCrossentropy(),
	metrics=['categorical_accuracy'])
history = model.fit(train, validation_data=test, epochs=10, steps_per_epoch=1410.375,
                    callbacks=model_checkpoint_callback)
classes = utils.get_imagenet_classes()
