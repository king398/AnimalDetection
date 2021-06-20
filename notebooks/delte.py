import shutil
import os
import tensorflow as tf

file = tf.io.gfile.glob("/content/train/" + '*/*/*.txt')

for i in file:
	print(i)
	os.remove(i)
