import shutil
import os
import tensorflow as tf
file = tf.io.gfile.glob(str("F:\Pycharm_projects\scientificProject\data"+'/*/*/Label'))



for i in file:
	shutil.rmtree(i)