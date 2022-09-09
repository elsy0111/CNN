from cgi import MiniFieldStorage
import numpy as np
import glob
import tensorflow as tf
import mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img_date = tf.io.read_file(mnist)
img_date = tf.io.decode_jpeg(img_date)
