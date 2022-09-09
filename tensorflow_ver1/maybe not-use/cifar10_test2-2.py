# cifar10を用いて画像認識を行い、正答率とlossを表示するプログラム.
# 作成日:2022/08/26, 変更:2022/08/27
# どうにかしてラベルの形式を変える方法を見つけたい！
# (今のところラベルを0, 1 の配列に変更するとshapeが変わってエラーになる)
# [参考]https://www.tensorflow.org/tutorials/images/cnn
#      https://www.tensorflow.org/tutorials/quickstart/advanced 
#      https://blog.shikoan.com/keras-multiple-label-output/
# [解説?]https://note.com/mlai/n/n64ca144365ac
#      https://note.nkmk.me/python-tensorflow-keras-basics/
# 参考資料よりコピペした内容を修正した。
# [kerasについて]https://www.tensorflow.org/guide/keras#model_subclassing
# できればこれを使用したい(Keras Functional API)--->https://www.tensorflow.org/guide/keras/functional/


import tensorflow as tf
#--->tensorflow version2.9.1
#import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import pandas
#mnist = tf.keras.datasets.mnist

#---------- 自作の損失関数(loss_function) ----------
# 今回はmnistの10のラベルを11に拡張(奇数か偶数かを追加)したものの例を使用したため、多少異なるかも
# デフォルトの損失関数だと、出力ユニットの数とラベルデータの数が等しい場合しか想定していない
# そのため、エラーが出る。つまり、損失関数の中で列方向のスライスをする必要がある。
# [解説]
# y_true_combinedという(バッチサイズ, 11)の形式で与えられるテンソルをスライスします。
# ちなみにy_predは(バッチサイズ, 10)の形式で返ってきます。
# スライスは基本的にはNumpyベースと変わりません。
# y_trueは(バッチサイズ, 10)という形式、is_oddは(バッチサイズ, )という形式にそれぞれなります。
# y_true_combinedはランク2であったのに、is_oddがランク1に落ちることがポイントです。
# 次に、keras.objectivesのcategorical_crossentropyを使い、通常のMNISTの分類と同じ交差エントロピーを計算します。
# keras.backendにもcategorical_crossentropyの関数はありますが、
# objectivesのほうは計算結果のランクが入力のランク-1される（つまりこの場合はランク1で返ってくる）、
# backendのほうはランクが落ちない（今回は使いませんでしたがランク2で返ってくる）という違いがあります。
# 今回はobjectivesの交差エントロピーを使っているので、is_oddとランク1のテンソル同士の積を計算しています。
# Kerasの損失関数は基本的にはサンプル単位で返してあげればあとは勝手にやってくれる
# (サンプル単位で荷重をかけたりすることが想定されているそうです）ので、
# サンプル単位で集計したランク1のテンソルを返す実装でOKです。
def loss_function(y_true_combined, y_pred):
    y_true, is_odd = y_true_combined[:, :10], y_true_combined[:, 10]
    return tf.keras.utils.to_categorical_crossentropy(y_true, y_pred) * is_odd

#*--------------- dataset読み込み ---------------*#
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("first y_train shape=", y_train.shape)
y_train = np.eye(10)[y_train].astype(np.float32)
print("now y_train shape=", y_train.shape)
y_test = np.eye(10)[y_test].astype(np.float32)
exit()


# 画像確認用
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    #plt.gray()   # 白黒での表示
    plt.grid(False)
    plt.imshow(x_train[i])
    # CIFARのlabelsはarrays(配列)である, 
    # そのため、追加のインデックスが必要。
    plt.xlabel(y_train[i] )
plt.show()

# データセットの調整?
# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")/255.0
x_test = x_test[..., tf.newaxis].astype("float32")/255.0

# [補足]tf.newaxisの中身はNone, 次元の追加を表しているだけ。
# (50000, 32, 32, 3)--->(50000, 32, 32, 3, 1)になった！(枚数, 縦(横), 横(縦), 高さ(BGR), 次元)
# x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0 でも同じかも


# y_train, y_test のデータ型はnumpyのndarray 
# ラベルデータをOne-hotベクトルに変更[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]の形式
##y_train = tf.keras.utils.to_categorical(y_train, 10)
##y_test = tf.keras.utils.to_categorical(y_test, 10)

#print(" 1.y_train:", type(y_train))
# y_train のデータ型はnumpy.ndarray
"""
# 前のやつでValluErrorになっていた原因の一つはこれかもしれない
#--->one-hot形式にすることで、引数の形式が変わっていた。そこを書き直す必要があるかも
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(y_train[1])
"""
# これは学習時に使用する。x_trainとy_train を一つのデータにまとめる
train_ds = tf.data.Dataset.from_tensor_slices(
  (x_train,
   y_train)
  ).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
  (x_test, 
   y_test)
  ).batch(32)
"""train_ds = tf.data.Dataset.from_tensor_slices(
  (x_train[..., tf.newaxis].astype("float32")/ 255.0, 
   tf.keras.utils.to_categorical(y_train, 10))
  ).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
  (x_test[..., tf.newaxis].astype("float32")/ 255.0, 
   tf.keras.utils.to_categorical(y_test, 10))
  ).batch(32)"""
# train_dsのデータ型はtensorflow.python.data.ops.dataset_ops.BatchDataset
#print(" 2.y_train:", type(y_train))
"""
# 確認用
for images, labels in train_ds:
  if(n<2):
    print("x_train:", images)
    print("y_train:", labels)
    n+=1
  else:
    break
"""

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
  # training=True は異なるレイヤーがある場合のみ必要
  # トレーニング中の動作と推論 (ドロップアウトなど).
    predictions = model(images, training=True)
    loss = loss_function(labels, predictions)  #loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(images, labels):
  # training=False は異なるレイヤーがない場合
  # トレーニング中の動作と推論 (ドロップアウトなど).
  predictions = model(images, training=False)
  t_loss = loss_function(labels, predictions)  #loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
#---------------------------------------------------
# 学習およびテストデータでの確かめ
# EPOCHS = 学習回数
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
