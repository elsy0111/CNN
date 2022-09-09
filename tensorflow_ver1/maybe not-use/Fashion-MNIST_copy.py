# kerasだとone-hot化の自動化が何処で行われているのかが不明なため、tensorflowをつかってみた
# p220まで書いた(途中で終わってます)
# 本を参考にFashion-MNISTを使用したtansorflowの機械学習のコードを書き写した
# 作成日：2022/09/01~

#---------- import ----------
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import mnist
#---------- read dataset ---------
(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()
"""
# 確認用
print(x_train.shape) # (60000, 28, 28)
print(t_train.shape) # (60000,)
print(x_test.shape)  # (10000, 28, 28)
print(t_test.shape)  # (10000,)

#---------- 50枚のデータを抽出してプロットしてみる ----------
# matplotlib inline
# ラベルに割り当てられたアイテム名を登録
class_names = ["0 T-shirt/top", "1 trouser", "2 pullover", "3 Dress", "4 Coat", 
               "5 Sandal", "6 Shirt", "7 Sneaker", "8 Bag", "9 Ankle boot"]

plt.figure(figsize = (25, 25))
# 訓練データから25枚の画像をプロットする
for i in range(25):
    # 5×5で出力
    plt.subplot(5, 5, i+1)
    # 縦方向の間隔をあける
    plt.subplots_adjust(hspace = 1.0)
    # 軸目盛を非表示にする
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # カラーマップをグレースケールに設定してプロット
    plt.imshow(x_train[i], cmap = plt.cm.binary)
    # x軸ラベルにアイテム名を出力
    plt.xlabel(class_names[t_train[i]])
plt.show()
"""
#---------- 画像データの正規化 ----------
# [重要] 多層パーセプトロンの入力層はベクトルの形(1階テンソルである必要がある)
# (60000, 28, 28)の訓練データを(60000, 784)の2階テンソルに変換
tr_x = x_train.reshape(-1, 784)
# 訓練データをfloat32(浮動小数点数)型にし、255で割ってスケール変換する
tr_x = tr_x.astype('float32') / 255
# (10000, 28, 28)のテストデータを(10000, 784)の2階テンソルに変換
ts_x = x_test.reshape(-1, 784)
# テストデータをfloat32(浮動小数点数)型にし、255で割ってスケール変換する
ts_x = ts_x.astype('float32') / 255
"""
# 確認用
print(tr_x.shape) # (60000, 784)
print(ts_x.shape) # (10000, 784)
"""
#---------- ラベルの処理 ----------
#print(t_train) # [9 0 0 ... 3 0 5]

# One-Hot化の処理
# クラスの数
class_num = 10 # 今回は10種類だから
# 訓練データの正解ラベルを変換
tr_t = keras.utils.to_categorical(t_train, class_num)
# テストデータの正解ラベルを変換
ts_t = keras.utils.to_categorical(t_test, class_num)
"""
# 確認用
print("tr_t[0]    :", tr_t[0])    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
print("tr_t.shape :", tr_t.shape) # (60000, 10)
print("ts_t[0]    :", ts_t[0])    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
print("ts_t.shape :", ts_t.shape) # (10000, 10)
"""
#---------- modelクラス定義(MLP) ----------
class MLP(keras.Model):
    # 多層パーセプトロン
    # Attributes:
    #  l1(Dense): 隠れ層
    #  l2(Dense): 出力層
    def __init__(self, hidden_dim, output_dim):
        # Parameters:
        #  hidden_dim(int): 隠れ層のユニット数(次元)
        #  output_dim(int): 出力層のユニット数(次元)
        
        super().__init__()
        # 隠れ層:活性化関数はrelu
        self.l1 = keras.layers.Dense(hidden_dim, activation = 'relu')
        # 出力層:活性化関数はsoftmax
        self.l2 = keras.layers.Dense(output_dim, activation = 'softmax')
        
    def call(self, x):
        # MLPのインスタンスからコールバックされる関数
        # Parameters: x(ndarray(float32)): 訓練データor検証データ
        # Returns(float32): MLPの出力として要素数3の1階テンソル
        h = self.l1(x)
        y = self.l2(h)
        return y

#---------- loss定義 ----------
# マルチクラス分類のクロスエントロピー誤差を求めるオブジェクト
cce = keras.losses.CategoricalCrossentropy()
def loss(t, y):
    # 損失関数
    # Parameters:
    #  t(ndarray(float32)):正解ラベル
    #  y(ndarray(float32)):予測値
    # Return: クロスエントロピー誤差
    return cce(t, y)    

#---------- 勾配降下アルゴリズムによるパラメーターの更新処理 ----------
# 勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = keras.optimizers.SGD(learning_rate = 0.1)
# 損失を記録するオブジェクトを生成