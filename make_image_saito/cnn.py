#* 初期値
#*----- import -----
from operator import imod
import tensorflow as tf
import keras.models as models
import keras.layers as layers
import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
#*----- import -----

dataset_num = 100  # 繰り返す回数

#* cnn で使用(格納用)
images = []  # 画像格納用
labels = []  # ラベル格納用

#*---------------------------------------------------

#*----- ここからCNN -----
# 画像サイズ
high = int(250)
width = int(250)

# データセットを分ける
train_images = []
train_labels = []
test_images = []
test_labels = []

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.3)

# 訓練用データ、テストデータに取り込んだデータを格納する
train_images = np.array(train_images)/255.0
train_labels = np.array(train_labels)
test_images = np.array(test_images)/255.0
test_labels = np.array(test_labels)


# 画像サイズなどの確認用
print("train_images.shape: ", train_images.shape) # (枚数, 縦, 横)
print("train_labels.shape: ", train_labels.shape) # (枚数,)
print("test_images.shape: ", test_images.shape) # (枚数, 縦, 横)
print("test_labels.shape: ", test_labels.shape) # (枚数,)
print("train_images[0]: ", train_images[0]) # ndarray(画像の要素)
print("train_labels[0]: ", train_labels[0]) # ndarray(ラベルの要素)

'''
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
'''
list_num = 88

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
model.add(layers.Dense(list_num))
model.summary()

model.compile(optimizer='adam', # パラメータの調整(学習方法)
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # 確率を使用したloss
              metrics=['accuracy'])
EPOCHS = 10
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
