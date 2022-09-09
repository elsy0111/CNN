# cifar10を普通の画像に置き換えてじっこうする(9/1~9/2)
# 変更点：グレースケールにするため、input_shapeを(32, 32, 3)から(32, 32, 1)に変更した
#        ラベルの数(44)に応じて、model.add(layers.Dense(44))に変更した(model定義の最後の部分softmax用)
#        テストデータは多いと結果がターミナルに出力されなくなるので10枚にしてます

# cifar10_copy1.pyにオリジナル有(ラベル変更済みはcifar10-changelabels.py)
# 画像の読み込み方法
# [参考]https://youtu.be/ThKRS7B5GFY
# [OpenCVを用いた画像の縮小方法]https://di-acc2.com/programming/python/18853/

# とりあえず、変数名はできる限り以前のデータから変えないようにしようと思う
# なんとなく、最初はJapanese_Allを使用します
#---------- import ----------
from operator import imod
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
x_train = []
test_images = []
# ラベルの保存用リスト
y_train = []
test_labels = []
#*---------- tensorflow_ver1/Japanese_Allの中の画像を読み込む ----------*#
# 画像のサイズ、各値の指定
high = int(480/4)
width = int(640/4)
#size = (high, width)
size = (width, high) # 縮小サイズ(ピクセル数)の指定

# tensorflow_ver1/*/*.png と指定すれば、ほかのファイルの中の画像を読み込むことが可能
n = 0 # カウンター用整数
for f in glob.glob("tensorflow_ver1/Japanese_All/*.png"):
    #print(f)
    # 画像ファイルの読み込み(3次元)
    img_data = cv2.imread(f)
    # サイズの変更
    img_data = cv2.resize(img_data, size)
    # １次元にする
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    #img_data = cv2.threshold(img_data, 0, 255, cv2.THRESH_OTSU)
    x_train.append(img_data)
    y_train.append(n)
    '''if n<10:
        test_images.append(img_data)
        test_labels.append(n)
    n += 1'''
    
f = glob.glob("tensorflow_ver1/Mel_Spectrogram_J01-20.png")
img_data = cv2.imread(f)
img_data = cv2.resize(img_data, size)
img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
test_images.append(img_data)
test_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #trainとtest, 英語と日本語など分ける場合は以下の指定をする
    ##省略
    
x_train = np.array(x_train) / 255.0
y_train = np.array(y_train)
#print(y_train)
#print(x_train[1])
print("ピクセル数(high, width, All): ", high, width, width*high)
#exit()
#*------------------------------------------------------------------------
# 訓練用データ、テストデータに取り込んだデータを格納する
train_images = x_train
train_labels = y_train
test_images = np.array(test_images)/255.0
test_labels = np.array(test_labels)
#test_images = x_train
#test_labels = y_train

# 画像サイズなどの確認用
print("train_images.shape: ", train_images.shape) # (枚数, 縦, 横)
print("train_labels.shape: ", train_labels.shape) # (枚数,)

# 画像の出力
plt.figure(figsize=(40,60))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(train_labels[i])
plt.show()

# ラベルの変換
train_labels = tf.keras.utils.to_categorical(train_labels, 44) #one-hot形式、float32になる
##test_labels = tf.keras.utils.to_categorical(test_labels, 44) #one-hot形式

# 今回だけとりあえずテストデータとトレーニングデータを一部、同じにした
#test_images = train_images
print("test_images.shape: ", test_images.shape)   # (枚数, 縦, 横)
#print("train_labels.shape: ", train_labels.shape) # (数値の数, 枚数)
#print("test_labels.shape: ", test_labels.shape)   # (数値の数, 枚数)
#print(test_labels[0])
#print(train_labels)

#exit()
#---------- 学習部分 ----------
# 画像処理(特徴を見つける)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(high, width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
# model.summary()
# ニューラルネットワーク
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(44))
model.summary()

model.compile(optimizer='adam', # パラメータの調整(学習方法)
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # 確率を使用したloss
              metrics=['accuracy'])
EPOCHS = 40
history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# 予測値
print("loss: CategoricalCrossentropy")
prediction = model.predict(test_images)
print(prediction)

exit()
for i in range(44):
    print(model.predict(test_images[i]))
