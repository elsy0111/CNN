# test1の書き直し
# cifar10 より、画像認識を行うプログラム。※学習に時間かかります。
# cifar10 は10種類の動物の画像(32*32ピクセル)が用意されたもの
# [参考] https://your-3d.com/python-picrecognition-cifar10/
#  [他]  https://qiita.com/takashi_42331/items/efc2039dc97bbf38b4ba
# [学習部分]https://colab.research.google.com/github/DeepInsider/playground-data/blob/master/docs/articles/tf2_keras_howtowrite.ipynb#scrollTo=Gl2fUM5N7kWG

# importする
from lib2to3.pgen2.pgen import PgenGrammar
import keras
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from keras import layers
import keras.backend as K
import pydot
import graphviz
from IPython.display import Image

#*--------------- dataset読み込み ---------------*#
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")
x_train = np.array(x_train)/255
x_test = x_test.astype("float32")
x_test = np.array(x_test)/255
#print("x_train:",x_train[0])
#print("x_test:", len(x_test[0]))
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#*--------------- model設計 ---------------*#
""" ↓ ??? """
# 正解率(精度)のカスタム指標(すべての書き方に共通)
def tanh_accuracy(y_true, y_pred):    # y_trueは正解、y_predは予測（出力）
  threshold = K.cast(0.0, y_pred.dtype)              # -1か1かを分ける閾値を作成
  y_pred = K.cast(y_pred >= threshold, y_pred.dtype) # 閾値未満で0、以上で1に変換
  # 2倍して-1.0することで、0／1を-1.0／1.0にスケール変換して正解率を計算
  return K.mean(K.equal(y_true, y_pred * 2 - 1.0), axis=-1)

# 活性化関数を変数（ハイパーパラメーター）として定義 ###
# 変数（モデル定義時に必要となる数値）
activation1 = layers.Activation(
    'tanh'                          # 活性化関数（隠れ層用）： tanh関数（変更可能）
    , name='activation1'            # 活性化関数にも名前付け
    )
activation2 = layers.Activation(
    'tanh'                          # 活性化関数（隠れ層用）： tanh関数（変更可能）
    , name='activation2'            # 活性化関数にも名前付け
    )
acti_out = layers.Activation(
    'softmax'                       # 活性化関数（出力層用）： softmax関数（固定）
    , name='acti_out'               # 活性化関数にも名前付け
    )

#---------------------------------------------------
# 定数（モデル定義時に必要となる数値）適当です
INPUT_FEATURES = 784  # 入力（特徴）の数： 28*28=784?(ピクセル数?)
LAYER1_NEURONS = 128     # ニューロンの数： 
LAYER2_NEURONS = 128     # ニューロンの数： 
OUTPUT_RESULTS = 10    # 出力結果の数： 10　(ラベルの数?)

# tf.keras.Modelをサブクラス化してモデルを定義
class NeuralNetwork(tf.keras.Model):
    ## レイヤーを定義 ##
    def __init__(self, *args, **kwargs):
        super(NeuralNetwork, self).__init__(*args, **kwargs)
        # 入力層は定義「不要」、実際の入力によって決まるので
        
        # 隠れ層：1つ目のレイヤー
        self.layer1 = layers.Dense(          # 全結合層
            #input_shape=(INPUT_FEATURES,),  # 入力層(定義不要)
            name='layer1',                   # 表示用に名前付け
            units=LAYER1_NEURONS)            # ユニットの数
         # 隠れ層：2つ目のレイヤー
        self.layer2 = layers.Dense(          # 全結合層
            name='layer2',                   # 表示用に名前付け
            units=LAYER2_NEURONS)            # ユニットの数
        # 出力層
        self.layer_out = layers.Dense(       # 全結合層
            name='layer_out',                # 表示用に名前付け
            units=OUTPUT_RESULTS)            # ユニットの数
        
    ## フォワードパスを定義 ##
    def call(self, inputs, training=None):  #入力と、訓練／評価モード
        #「出力=活性化関数(第n層(入力))」の形式で記述
        x1 = activation1(self.layer1(inputs))
        x2 = activation2(self.layer2(x1))
        outputs = acti_out(self.layer_out(x2))
        return outputs
    
model = NeuralNetwork()                # モデルの生成

# これは確認用、なくてもいい。
# 計算グラフなしでは[model.summary()]は失敗する(ValueError)
model3 = NeuralNetwork()                # モデルの生成

# 「仮のモデル」をFunctional APIで生成する独自関数
def get_functional_model(model):
  # このコードは、「リスト3-1のFunctional API」とほぼ同じ
  x = layers.Input(shape=(INPUT_FEATURES,), name='layer_in')
  temp_model = tf.keras.Model(
      inputs=[x],
      outputs=model.call(x),  # ※サブクラス化したモデルの`call`メソッドを指定
      name='subclassing_model3')  # 仮モデルにも名前付け
  return temp_model

# Functional APIの「仮のモデル」を取得
f_model = get_functional_model(model3)

# モデルの内容を出力
f_model.summary()

# モデルの構成図を表示
tf.keras.utils.plot_model(f_model, show_shapes=True, show_layer_names=True, to_file='tensorflow_ver1\model.png')
Image(retina=False, filename='tensorflow_ver1\model.png')

#ValueErrorの発生、修正が必要(compileかfitがダメっぽい)
model.compile(tf.keras.optimizers.SGD(learning_rate=0.03), 'categorical_crossentropyS', metrics=["accuracy"])
#model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
'''
InvalidArgumentError の発生 ##ValueErrorになった
https://qiita.com/JarvisSan22/items/80b1e46a3164f430989f
'''
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=1024, epochs=5, verbose=1)
#hist = model.fit(x_train, y_train, batch_size=32, epochs=5)  # epochs=6以降はほぼ結果は変わらなかった
model.predict([[0.1,-0.2]])
#---------------------------------------------------
import matplotlib.pyplot as plt

# 学習結果（損失）のグラフを描画
train_loss = hist.history['loss']
valid_loss = hist.history['val_loss']
epochs = len(train_loss)
plt.plot(range(epochs), train_loss, marker='.', label='loss (Training data)')
plt.plot(range(epochs), valid_loss, marker='.', label='loss (validation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
#---------------------------------------------------
# 次のコードのためにモデル（重みやバイアス）をリセットする
tf.keras.backend.clear_session() # 計算グラフを破棄する
del model                        # 変数を削除する

model = NeuralNetwork()          # モデルの再生成
#---------------------------------------------------         
"""# カスタムの評価関数を実装（TensorFlow低水準API利用）
# （tf.keras.metrics.binary_accuracy()の代わり）
def tanh_accuracy(y_true, y_pred):           # y_trueは正解、y_predは予測（出力）
  threshold = K.cast(0.0, y_pred.dtype)              # -1か1かを分ける閾値を作成
  y_pred = K.cast(y_pred >= threshold, y_pred.dtype) # 閾値未満で0、以上で1に変換
  # 2倍して-1.0することで、0/1を-1.0/1.0にスケール変換して正解率を計算
  return K.mean(K.equal(y_true, y_pred * 2 - 1.0), axis=-1)
"""
#---------------------------------------------------
# カスタムの評価関数クラスを実装（サブクラス化）
# （tf.keras.metrics.BinaryAccuracy()の代わり）
class TanhAccuracy(tf.keras.metrics.Mean):
  def __init__(self, name='tanh_accuracy', dtype=None):
    super(TanhAccuracy, self).__init__(name, dtype)

  # 正解率の状態を更新する際に呼び出される関数をカスタマイズ
  def update_state(self, y_true, y_pred, sample_weight=None):
    matches = tanh_accuracy(y_true, y_pred)
    return super(TanhAccuracy, self).update_state(
        matches, sample_weight=sample_weight)

   

