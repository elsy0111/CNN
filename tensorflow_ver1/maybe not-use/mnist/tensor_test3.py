# mnistを使用し、1~9までの手書き数字を予測するプログラム
# [参考資料]
# https://youtu.be/I-JtV2CNWlM
# https://youtu.be/ThKRS7B5GFY
# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ja

# tensorflow_version 2.9.1
#import glob
from re import X
import tensorflow as tf
from keras import backend
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
#アプリ作成用に必要なやつ⇒import gradio as gr
tf.get_logger().setLevel("ERROR")

#*------------------- ここから -------------------*#
# mnistのデータを読み込む(train用, test用)
# 学習データ -- 60000枚ある(?)
# テストデータ -- 10000枚ある(?)
# x_ -- 画像データ ⇒ 色があるから0 ~ 255までのデータになってる
# y_ -- ラベル(0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# numpyの形式へ変更、x_は0~1の数値に収めるために256で割る。⇒これにより、正答率が上がった(0.05くらい)。
x_train = np.array(x_train)/255.0
y_train = np_utils.to_categorical(y_train)
x_test = np.array(x_test)/255.0
y_test = np_utils.to_categorical(y_test)
#print(y_train[0])



# 画像データとラベルの要素数を表示 ⇒ 一つ一つ表示されます
print("画像データの要素数", x_train.shape)
print("ラベルデータの要素数", y_train.shape)

# ラベルと画像データを表示
for i in range(0,10):
    print("ラベル", y_train[i])
    plt.imshow(x_train[i].reshape(28, 28), cmap='Greys')
    plt.show()
    

# モデルの作成
from keras.models import Sequential
from keras.layers.core import Dense, Activation
 
model = Sequential()
model.add(Dense(16, input_shape=(64,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(accuracy)

loss_cnt = 0
size = len(x_test)
 
for i in range(size):
    m_predict = model.predict_classes(x_test[i:i+1], batch_size=1)
 
    if y_test[i][m_predict]:
        print('{}: 正解'.format(i))
    else:
        print('{}: 不正解'.format(i) + ' 予測:{0}, 正解:{1}'.format(m_predict, y_test[i]))
        loss_cnt+=1
     
print('正解率: {}'.format((size - loss_cnt) / size))
"""model = tf.keras.models.Sequential([
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
#model.summary()
# ⇒入力層 -- 784, 中間層 -- 128, 出力層 -- 10
# fit -- 学習
# epochs -- 学習の回数（周回する回数を指定）
print("学習開始(trainを何回か実行)")
model.fit(x_train, y_train, epochs=8)

# モデルの評価
print("学習の評価(test実行)")
model.evaluate(x_test, y_test)
"""

"""
# 確認用（予測と結果を表示させたい）
# 予測されたものの表示方法がわからん笑←これないと意味ない
number = 100   # なんか好きな数字入れる
print("ラベル", y_test[number])
plt.imshow(x_test[number].reshape(28, 28), cmap='Greys')
plt.show()
"""
