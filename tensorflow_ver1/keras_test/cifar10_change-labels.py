# まずはlabel変更に取り掛かる(8/31~9/1)
# 変更点：train_labels, test_labelsをone-hot形式に手動で変更、
#        それに伴って、lossのSparseCategoricalCrossentropyの"Sparse"の部分のみ削除
#        epockの数を10から5に変更しました(時間短縮のため)
# [参考]https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/one-hot

# ひとつずつ変更を加える--->エラーの原因追及のため
# cifar10_copy1にオリジナル有
# [参考資料]https://www.tensorflow.org/tutorials/images/cnn

from operator import imod
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
#---------- datesetの読み込み ----------
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# ピクセル値を 0 ~ 1 の範囲に正規化する
train_images, test_images = train_images / 255.0, test_images / 255.0

# 画像の出力
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

print("train_labels shape=",train_labels.shape)#--->(50000, 1)
print("train_labels[0]",train_labels[0])#--->6
# ---------- labelの変更を行う ------------
train_labels = tf.keras.utils.to_categorical(train_labels, 10) #one-hot形式、float32になる
#train_labels = np.array(train_labels, dtype = int)#.tolist()   # int型にする(.tolistはlistの形式にすることが可能)
#train_labels = np.array(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels, 10) #one-hot形式
#test_labels = np.array(test_labels, dtype = int)#.tolist()   # int型にする(.tolistはlistの形式にすることが可能)

# 確認用
print("new-train_labels shape=",train_labels.shape)#--->(50000, 10)
print("new-train_labels[0]:",train_labels[0])#--->[0 0 0 0 0 0 1 0 0 0]
print("new-train_labels[0] type:",type(train_labels[0]))#---><class 'numpy.ndarray'>
#print("train_images.shape: ", train_images.shape)
#exit()

# 画像処理(特徴を見つける)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.summary()
# ニューラルネットワーク
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam', # パラメータの調整(学習方法)
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # 確率を使用したloss
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)