# mnistを使用し、1~9までの手書き数字を予測するプログラム(カスタム有り)
# [参考資料]
# https://atmarkit.itmedia.co.jp/ait/articles/2003/23/news024.html
# tensorflow_version 2.9.1を使用
import tensorflow as tf
import keras
from keras.datasets import mnist
import keras.backend as K  # import tensorflow.keras.backend as K
from re import X
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from keras import layers
tf.get_logger().setLevel("ERROR")

#*------------------- ここから -------------------*#
# mnistのデータを読み込む(train用, test用)
# 学習データ -- 60000枚ある(?)
# テストデータ -- 10000枚ある(?)
# x_ -- 画像データ ⇒ 色があるから0 ~ 255までのデータになってる
# y_ -- ラベル(0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# numpyの形式へ変更、x_は0~1の数値に収めるために255.0で割る。
x_train = np.array(x_train)/255.0
y_train = np.array(y_train)
x_test = np.array(x_test)/255.0
y_test = np.array(y_test)

# ラベルデータをOne-Hot形式に変更(例)：[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#y_train = keras.utils.to_categorical(y_train, 10)
#y_test = keras.utils.to_categorical(y_test, 10)
#print(y_test[0])

#*---------- ここからカスタムなど ----------*#
# 詳しくは 'custom_example'を参照
# カスタムの活性化関数のPython関数を実装
def custom_activation(x):
    return (tf.exp(x)-tf.exp(-x))/(tf.exp(x)+tf.exp(-x))
"""
# カスタムの活性化関数のPython関数を実装(kerasのbackendバージョン)
def custom_activation(x):
    return (K.exp(x)-K.exp(-x))/(K.exp(x)+K.exp(-x))
"""
# カスタムの活性化関数クラスを実装（レイヤーのサブクラス化）
# （tf.keras.layers.Activation()の代わり）
class CustomActivation(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(CustomActivation, self).__init__(**kwargs)

  def call(self, inputs):
    return custom_activation(inputs)
#-----------------------------------------------------
# カスタムの全結合層（線形変換）のPython関数を実装
def fully_connected(inputs, weights, bias):
  return tf.matmul(inputs, weights) + bias

# カスタムのレイヤークラスを実装（レイヤーのサブクラス化）
# （tf.keras.layers.Dense()の代わり）
class CustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, input_dim=None, **kwargs):
    self.input_dim = input_dim  # 入力の次元数（＝レイヤーの入力数）
    self.units = units          # ニューロン数（＝レイヤーの出力数）
    super(CustomLayer, self).__init__(**kwargs)

  def get_config(self):
    # レイヤー構成をシリアライズ可能にするメソッド（独自の設定項目がメンバー変数としてある場合など、必要に応じて実装する）
    config = super(CustomLayer, self).get_config()
    config.update({
        'input_dim': self.input_dim,
        'units': self.units
    })
    return config

  def build(self, input_shape):
    #print(input_shape) # 入力形状。例えば「(2, 2)」＝2行2列なら入力の次元数は2列
    input_data_dim = input_shape[-1] # 入力の次元数（＝レイヤーの入力数）

    # 入力の次元数をチェック（デバッグ）
    if self.input_dim != None:
      assert input_data_dim == self.input_dim  # 指定された入力次元数と実際の入力次元数が異なります

    # 重みを追加する
    self.kernel = self.add_weight(
        shape=(input_data_dim, self.units),
        name='kernel',
        initializer='glorot_uniform',  # 前々回のリスト1-3のような独自の関数も指定できる
        trainable=True)

    # バイアスを追加する
    self.bias = self.add_weight(
        shape=(self.units,),
        name='bias',
        initializer='zeros',
        trainable=True)
    
    self.built = True # Layerクラスでビルド済みかどうかを管理するのに使われている（なくても大きな問題はない）
    super(CustomLayer, self).build(input_shape) # 上と同じ意味。APIドキュメントで推奨されている書き方

  def call(self, inputs):
    return fully_connected(inputs, self.kernel, self.bias)
#-----------------------------------------------------
# カスタムの最適化アルゴリズムクラスを実装（オプティマイザのサブクラス化）
# （tf.keras.optimizers.SGDの代わり）
class CustomOptimizer(tf.keras.optimizers.Optimizer):
  def __init__(self, learning_rate=0.01, name='CustomOptimizer', **kwargs):
    super(CustomOptimizer, self).__init__(name, **kwargs)
    self.learning_rate = kwargs.get('lr', learning_rate)

  def get_config(self):
    config = super(CustomOptimizer, self).get_config()
    config.update({
        'learning_rate': self.learning_rate
    })
    return config

  def _create_slots(self, var_list):
    for v in var_list: # `Variable`オブジェクトのリスト
      self.add_slot(v, 'accumulator', tf.zeros_like(v))
    # 参考実装例。※ここで作成したスロットは未使用になっている

  def _resource_apply_dense(self, grad, var):
    # 引数「grad」： 勾配（テンソル）
    # 引数「var」： 更新対象の「変数」を示すリソース（“resource”データ型のテンソル）
    # var.device の内容例： /job:localhost/replica:0/task:0/device:CPU:0
    # var.dtype.base_dtype の内容例： <dtype: 'float32'>
    acc = self.get_slot(var, 'accumulator') # 参考実装例（スロットは未使用）
    return var.assign_sub(self.learning_rate * grad)  # 変数の値（パラメーター）を更新
    
  def _resource_apply_sparse(self, grad, var, indices):
    # 引数「grad」： 勾配（インデックス付きの、スパースなテンソル）
    # 引数「var」： 更新対象の「変数」を示すリソース（“resource”データ型のテンソル）
    # 引数「indices」： 勾配がゼロではない要素の 「インデックス」（整数型のテンソル）
    raise NotImplementedError("今回は使わないので未実装")
    # return ……変数の値（パラメーター）を更新する「操作」を返却する……
#-----------------------------------------------------
# カスタムの損失関数のPython関数を実装
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# カスタムの損失関数クラスを実装（レイヤーのサブクラス化）
# （tf.keras.losses.MeanSquaredError()の代わり）
class CustomLoss(tf.keras.losses.Loss):
  def __init__(self, name="custom_loss", **kwargs):
    super(CustomLoss, self).__init__(name=name, **kwargs)

  def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)  # 念のためTensor化
    y_true = tf.cast(y_true, y_pred.dtype) # 念のため同じデータ型化
    return custom_loss(y_true, y_pred)
#-----------------------------------------------------
# 正解かどうかを判定する関数を実装
def custom_matches(y_true, y_pred):         # y_trueは正解、y_predは予測（出力）
  threshold = tf.cast(0.0, y_pred.dtype)              # -1か1かを分ける閾値を作成
  y_pred = tf.cast(y_pred >= threshold, y_pred.dtype) # 閾値未満で0、以上で1に変換
  # 2倍して-1.0することで、0／1を-1.0／1.0にスケール変換して正解率を計算
  return tf.equal(y_true, y_pred * 2 - 1.0) # 正解かどうかのデータ（平均はしていない）

# カスタムの評価関数クラスを実装（サブクラス化）
# （tf.keras.metrics.BinaryAccuracy()の代わり）
class CustomAccuracy(tf.keras.metrics.Mean):
  def __init__(self, name='custom_accuracy', dtype=None):
    super(CustomAccuracy, self).__init__(name, dtype)

  # 正解率の状態を更新する際に呼び出される関数をカスタマイズ
  def update_state(self, y_true, y_pred, sample_weight=None):
    matches = custom_matches(y_true, y_pred)
    return super(CustomAccuracy, self).update_state(
        matches, sample_weight=sample_weight) # ※平均は内部で自動的に取ってくれる
#-----------------------------------------------------

activation1 = CustomActivation(name='activation1')
activation2 = CustomActivation(name='activation2')
acti_out = layers.Activation('softmax', name='acti_out')
 
# それぞれの数値を指定(適当)
LAYER1_NEURONS = 128
LAYER2_NEURONS = 64
OUTPUT_RESULTS = 10
# モデルの設計
class NeuralNetwork(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super(NeuralNetwork, self).__init__(*args, **kwargs)
    self.layer1 = CustomLayer(units=LAYER1_NEURONS, name='layer1')
    self.layer2 = CustomLayer(units=LAYER2_NEURONS, name='layer2')
    self.layer_out = CustomLayer(units=OUTPUT_RESULTS, name='layer_out')
  def call(self, inputs, training=None):
    x1 = activation1(self.layer1(inputs))
    x2 = activation2(self.layer2(x1))
    outputs = acti_out(self.layer_out(x2))
    return outputs

# モデルの生成
model = NeuralNetwork()

# モデルの概要と構成図の出力
def get_functional_model(model):
  x = layers.Input(shape=(28,), name='layer_in')
  temp_model = tf.keras.Model(inputs=[x], outputs=model.call(x), name='custom_model')
  return temp_model
f_model = get_functional_model(model)
f_model.summary()
tf.keras.utils.plot_model(f_model, show_shapes=True, show_layer_names=True, to_file='model.png')
#from IPython.display import Image
#Image(retina=False, file='..\model.png')

# モデルの学習
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
try:
    model.fit(x_train, y_train, epochs=8)
    model.evaluate(x_test, y_test)

except ValueError as error:
  print(error)

'''
# モデルの作成
model = tf.keras.models.Sequential([
    # 入力層(28*28ピクセル?)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 中間層
    tf.keras.layers.Dense(128, activation='relu'),
    # 隠れ層(割合の変更の必要がある,なくてもいい)
    tf.keras.layers.Dropout(0.13), 
    # 出力層(10は 0~9の10種類に)
    tf.keras.layers.Dense(10, activation='softmax')
])

# モデルの学習
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
# fit -- 学習
# epochs -- 学習の回数（周回する回数を指定）
print("学習開始(trainを何回か実行)")
model.fit(x_train, y_train, epochs=8)

# モデルの評価
print("学習の評価(test実行)")
model.evaluate(x_test, y_test)
'''
"""
# 確認用（予測と結果を表示させたい）
# 予測されたものの表示方法がわからん笑←これないと意味ない
number = 100   # なんか好きな数字入れる
print("ラベル", y_test[number])
plt.imshow(x_test[number].reshape(28, 28), cmap='Greys')
plt.show()
"""
