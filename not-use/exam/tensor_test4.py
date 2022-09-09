# [参考]https://aizine.ai/tensorflow-python0722/

# import文
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
# mnistの読み込み
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train)/255.0, np.array(x_test)/255
y_train, y_test = np.array(y_train), np.array(y_test)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=1)

"""from __future__ import absolute_import, division, print_function, unicode_literals
from sre_parse import FLAGS
import tensorflow as tf
from keras import backend
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

tf.get_logger().setLevel("ERROR")

# mnistのデータを読み込む(train用, test用)
# x_ -- 画像データ, y_ -- ラベル(0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# numpyの形式へ変更、x_は0~1の数値に収めるために255で割る。⇒これにより、正答率が上がった(0.05くらい)。
x_train = np.array(x_train)/255.0
y_train = np.array(y_train)
x_test = np.array(x_test)/255.0
y_test = np.array(y_test)

# パラメータの付与を可能にする : FLAGS = tf.app.flags.FLAGS
# とりあえず値は適当
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary-dir', '/summary', 
                           "TensorBoard用のログを出力するディレクトリのパス")
tf.app.flags.DEFINE_integer('max-epoch', 5, "最大学習エポック数")
tf.app.flags.DEFINE_integer('bach-size', 1000, "1回のトレーニングステップに用いるデータのバッチサイズ")
tf.app.flags.DEFINE_float('learning-rate', 0.07, "学習率")
tf.app.flags.DEFINE_integer('test-date', 303, "テスト用データの数")
tf.app.flags.DEFINE_integer('training-date', 10000, "学習用データの数")
tf.app.flags.DEFINE_boolean('skip-training', False, "学習をスキップしてテストだけする場合は指定(True)")

#---------- ニューラルネットワーク ----------#
# 入力部分
#with tf.name_scope("x_test"):
    

"""