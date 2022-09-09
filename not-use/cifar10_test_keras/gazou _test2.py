# cifar10 より、画像認識を行うプログラム。※学習に時間かかります。
# cifar10 は10種類の動物の画像(32*32ピクセル)が用意されたもの
# [参考] https://your-3d.com/python-picrecognition-cifar10/
#  [他]  https://qiita.com/takashi_42331/items/efc2039dc97bbf38b4ba
# import
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
# dateset read
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# ラベルデータをOne-Hot形式に変更(例)：[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = np.array(x_train)/255
y_train = np.array(y_train)
x_test = np.array(x_test)/255
y_test = np.array(y_test)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#print(x_train[0])
#print(y_test[0])

model = tf.keras.models.Sequential([
    # 入力層(ピクセル数?)
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    # 中間層
    tf.keras.layers.Dense(128, activation='relu'),
    # 隠れ層(割合の変更の必要がある,なくてもいい)
    #tf.keras.layers.Dropout(0.13), 
    # 出力層(10は 0~9の10種類に)
    tf.keras.layers.Dense(
        units=64,
        activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax')
])

"""#---------- modelの定義 ----------#
# Sequential：ただ層を積み上げるだけの単純なモデル、keras
model = Sequential()
model.add(Conv2D(32,(3,3), padding="same", input_shape=(32,32,3)))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))
"""
#--------- modelのコンパイル ----------#
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#--------- 学習の実行 ----------#
hist = model.fit(
    x_train, y_train,
    batch_size=32, 
    epochs=4,     # 回数
    verbose=1,    # ログの設定？
    validation_data=(x_test, y_test))  # 検証データ

#--------- 評価 ----------#
score = model.evaluate(x_test, y_test, verbose=1) # 検証データでの accuracy, lossの代入
print("accuracy=", score[1], "loss", score[0])

"""
#--------- modelの保存 ----------#
model.save("ファイル名.h5") #モデルデータ保存
model.save_weights("ファイル名.h5") #重みデータ保存
"""

