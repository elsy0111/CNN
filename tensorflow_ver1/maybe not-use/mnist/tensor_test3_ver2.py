# mnistを使用し、1~9までの手書き数字を予測するプログラム
# [参考資料]
# https://youtu.be/I-JtV2CNWlM
# https://youtu.be/ThKRS7B5GFY
# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ja
# tensorflow_version 2.9.1

#import glob  # 画像をディレクトリから読み込む際に使用する
from re import X
import tensorflow as tf
from keras import backend
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

tf.get_logger().setLevel("ERROR")

#*-------------- 画像データの読み込みなど --------------*#

# mnistのデータを読み込む(train用, test用)
# x_ -- 画像データ, y_ -- ラベル(0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.eye(10)[y_train].astype(np.float32)
y_test = np.eye(10)[y_test].astype(np.float32)

# print(x_train)            4 demention array
# print(y_train)            1 demention array
# print(len(x_train))       60000 datasets
# print(len(y_train))       60000 datasets
# print(len(x_train[1]))    28 pixels array
# print(len(x_train[1][1])) 28 pixels array

#class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(type(y_train))
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #plt.gray()   # 白黒での表示
    plt.imshow(x_train[i])
    # CIFARのlabelsはarrays(配列)である, 
    # そのため、追加のインデックスが必要。
    plt.xlabel(y_train[i])
plt.show()
x_train = (x_train.reshape(-1, 784) / 255.0).astype(np.float32)
x_test = (x_test.reshape(-1, 784) / 255.0).astype(np.float32)

"""# numpyの形式へ変更、x_は0~1の数値に収めるために255で割る。⇒これにより、正答率が上がった(0.05くらい)。
x_train = np.array(x_train)/255
y_train = np.array(y_train)
x_test = np.array(x_test)/255
y_test = np.array(y_test)
"""
#*------------------- モデルの作成 -------------------*#

# 定数(モデル定義時に必要となる数値)
IMPUT_FEATURES = (28, 28, ) #入力(特徴)の数
LAYER1_NEURONS = 128      #ニューロンの数
LAYER2_NEURONS = 64       #ニューロンの数
OUTPUT_RESULTS = 10       #出力結果の数

# パラメータ(ニューロンへの入力で必要となるもの)
weight_array = np.array([[1],     # 重み
                         [1]])   
bias_array = np.array([1])        # バイアス

# 積層型のモデルの定義
model = tf.keras.models.Sequential([
    # 入力層
    tf.keras.layers.Flatten(
        input_shape=IMPUT_FEATURES),
    # 隠れ層：1つ目のレイヤー
    tf.keras.layers.Dense(
        units=LAYER1_NEURONS,
        activation='relu'),
    # 隠れ層：2つ目のレイヤー
    tf.keras.layers.Dense(
        units=LAYER2_NEURONS,
        activation='tanh'),
    # 出力層
    tf.keras.layers.Dense(
        10, activation='softmax')
])

"""
# 特に指定しない場合
model = tf.keras.models.Sequential([
    # 入力層(28*28ピクセル?)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 中間層
    tf.keras.layers.Dense(128, activation='relu'),
    # 隠れ層？(割合の変更の必要がある,なくてもいい)
    tf.keras.layers.Dropout(0.13), 
    # 出力層(10は 0~9の10種類に)
    tf.keras.layers.Dense(10, activation='softmax')
])
"""

# モデルの学習
# sparse_categorical_crossentropy
# ---> (mnistのように)多クラス分類問題の正解データをカテゴリ番号で与えている場合に使用
model.compile(
    optimizer='adam',                       # 最適化：勾配も指数移動平均で使う(らしい)
    loss='sparse_categorical_crossentropy', # 損失関数：
    metrics=['accuracy']                    # 精度
)
model.summary()
# --->各層の確認（入力層：784, 中間層：128, 出力層：10）
# model.fit：学習, epochs：学習の回数（周回する回数を指定）
print("学習開始(trainを何回か実行)")
model.fit(x_train, y_train, epochs=5)  # epochs=6以降はほぼ結果は変わらなかった

# モデルの評価
print("学習の評価(test実行)")
model.evaluate(x_test, y_test)

"""
# 確認用（予測と結果を表示させたい）
# 予測されたものの表示方法がわからん笑←これないと意味ない
number = 100   # なんか好きな数字入れる
print("ラベル", y_test[number])
plt.imshow(x_test[number].reshape(28, 28), cmap='Greys')
plt.show()
"""