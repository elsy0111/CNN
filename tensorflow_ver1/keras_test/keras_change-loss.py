# loss変更用
# テストデータ、数値などを変更してより良い結果が得られるように考察する用(9/2~ )
# 変更点：(まだしてないです)テストデータを01～20を重ねた画像に変更しました

# [予測値出力]この動画の後半https://www.youtube.com/watch?v=ThKRS7B5GFY&list=PLYn3_Z-PgUCMtMva-rayHe-HsWQuqx2FJ&index=5
# [関数など]https://qiita.com/code0327/items/4f8f656bed23140b8962
# [lossの種類]https://www.tensorflow.org/api_docs/python/tf/keras/losses
# [出力層、活性化関数、損失関数]https://qiita.com/pocokhc/items/d67b63ec9ca74b453093
# cifar10_change-images.pyにオリジナル有
# なんとなく、Japanese_Allを使用します
#---------- import ----------
from operator import imod
from typing import OrderedDict
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib
#----- 画像読み込みに必要なimport -----
from re import X
from tkinter import N
#import numpy as np
#import tensorflow as tf
import glob
from PIL import Image
import cv2

#---------- datesetの読み込み ----------
# 画像(数値データ)の保存用リスト
train_images = []
test_images = []
# ラベルの保存用リスト
train_labels = []
test_labels = []
#---------- tensorflow_ver1/Japanese_Allの中の画像を読み込む ----------#
# 画像のサイズ、各値の指定(整数じゃないとエラーが出る)
bunbo = 4
high = int(480/bunbo)
width = int(640/bunbo)
size = (width, high) # 縮小サイズ(ピクセル数)の指定

# tensorflow_ver1/*/*.png と指定すれば、ほかのファイルの中の画像を読み込むことが可能
n = 0 # カウンター用整数
for f in glob.glob("tensorflow_ver1/Japanese_All/*.png"):
    #print(f)
    # 画像ファイルの読み込み(3次元)
    img_data = cv2.imread(f)
    # サイズ, ピクセル数の変更
    #img_data = img_data[60:425, 80:575]
    img_data = cv2.resize(img_data, size)
    # １次元にする
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    #img_data = cv2.threshold(img_data, 0, 255, cv2.THRESH_OTSU)
    train_images.append(img_data)
    train_labels.append(n)
    # テストデータの作成(今回は0~9までのデータを使用)
    if n==0:
        test_images.append(img_data)
        test_labels.append(n)
    n += 1
    #trainとtest, 英語と日本語など分ける場合は以下の指定をする
    ##省略
    
# 確認用
print("ピクセル数(high, width, All): ", high, width, width*high)
#exit()

#---------- データを整理する(正規化など) ----------
train_images = np.array(train_images)/255.0
train_labels = np.array(train_labels)
test_images = np.array(test_images)/255.0
test_labels = np.array(test_labels)
#test_images = train_images
#test_labels = train_labels

# 画像サイズなどの確認用
#print("train_images.shape: ", train_images.shape) # (枚数, 縦, 横)
#print("train_labels.shape: ", train_labels.shape) # (枚数,)
'''
#----- 画像の出力 -----
plt.figure(figsize=(40,60))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()
'''
#exit()
#----- ラベルの変換 -----
train_labels = tf.keras.utils.to_categorical(train_labels, 44) #one-hot形式、float32になる
test_labels = tf.keras.utils.to_categorical(test_labels, 44) #one-hot形式

# 今回だけとりあえずテストデータとトレーニングデータを一部、同じにした
#test_images = train_images
# 確認用
print("test_images.shape: ", test_images.shape)   # (枚数, 縦, 横)
#print("train_labels.shape: ", train_labels.shape) # (数値の数, 枚数)
#print("test_labels.shape: ", test_labels.shape)   # (数値の数, 枚数)
#print(test_labels[0])
#print(train_labels)

#exit()
#---------- 学習部分 ----------
# 画像処理(特徴を見つける)
# 活性化関数: relu
# layers: Conc2D, MaxPooling2D
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
          input_shape=(high, width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
"""# 隠れ層増やしてみた
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
"""
# model.summary()
#----- ニューラルネットワーク -----
# layers.Flatten: 数値を1次元にしてる(画像を線にするイメージ?)
# 活性化関数: relu
# 数値の収束に'softmax'を使用してる(省略してます)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dense(16, activation='relu'))
#model.add(layers.Dense(16, activation='relu'))
#model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(44, activation='sigmoid'))
# パラメーターなどの表示(省略可)
model.summary()
#exit()
# [optimizerの種類]https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# lossの指定
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # 確率を使用したloss
#loss=tf.keras.losses.Huber(), # 全く合わないので没
#----- compile -----
model.compile(optimizer='adam',
              #optimizer='sgd', 
              loss=loss,
              metrics=['accuracy']
              )
#----- 学習開始 -----
EPOCHS = 40   # 学習回数の指定
history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels))
#----- 機械の予測の出力 -----
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#----- テスト結果の出力 -----
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(" ")
print("test_accuracy: ", test_acc)
print("test_loss:     ", test_loss)
# 予測値
print("loss: ", type(loss))
prediction = model.predict(test_images)
"""
prediction = prediction.tolist()
prediction = np.array(prediction, dtype='float')
"""
print(prediction)
exit()

# ↓適当に書いたけどできなかった
for i in range(44):
    print(model.predict(test_images[i]))
