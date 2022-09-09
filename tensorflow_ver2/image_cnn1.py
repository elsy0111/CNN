# cifar10を用いて画像認識を行い、正答率とlossを表示するプログラム.
# 作成日:2022/08/30
# [参考]https://www.tensorflow.org/tutorials/images/cnn
#       https://www.tensorflow.org/tutorials/quickstart/advanced
# 参考資料よりコピペした内容を修正した。
# [kerasについて]https://www.tensorflow.org/guide/keras#model_subclassing
# できればこれを使用したい(Keras Functional API)--->https://www.tensorflow.org/guide/keras/functional/

#[推測]88個のリスト(?)を88個の0, 1の並びの数字に変換すればそのまま使用できるのでは？
#[例][0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, ...]--->00000110100...
#確率とかおかしくなっちゃうかな...?
import tensorflow as tf
#--->tensorflow version2.9.1
#import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

#*--------------- dataset読み込み ---------------*#
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0
# データセットの調整?
# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")/255.0
x_test = x_test[..., tf.newaxis].astype("float32")/255.0
#print(x_train.shape)#--->(50000, 32, 32, 3)
# ラベルをone-hot形式へ変換
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
"""
# 前のやつでValluErrorになっていた原因の一つはこれかもしれない
#--->one-hot形式にすることで、引数の形式が変わっていた。そこを書き直す必要があるかも
#print(y_train[0])#--->5
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(y_train[0])
"""
"""# 画像確認用
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #plt.gray()   # 白黒での表示
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(y_train[i][0])
plt.show()
"""

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#*--------------- model設計 ---------------*#
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='tanh')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#---------------------------------------------------
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
#---------------------------------------------------
EPOCHS = 3

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )