# [参考]https://chusotsu-program.com/tensorflow-keras-digits/
# mnist を one-hot形式で機械学習する方法を見つける
# 問題画像、ラベルの出力まではできたが、やはり学習部分でエラーが出る。
import tensorflow as tf
#--->tensorflow version2.9.1
#import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
# データセットの調整
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0
x_train = x_train[..., tf.newaxis].astype("float32") / 255.0 # 0~1の数値の行列
x_test = x_test[..., tf.newaxis].astype("float32") / 255.0
print(y_train[0], type(y_train[0]))

y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
y_test = np.eye(10)[y_test].astype(np.float32)
y_test = np.array(y_test)
print(y_train[0], type(y_train[0]))
print(y_test[0], type(y_test[0]))
print(x_train[0], end=" ")

#y_train = np.eye(10)[y_train].astype(np.float32)   # ラベルをone-hot形式へ変更
#y_test = np.eye(10)[y_test].astype(np.float32)
# 画像確認用
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #plt.gray()   # 白黒での表示
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(y_train[i])
plt.show()

# modelの作成と学習
from keras.models import Sequential
from keras.layers.core import Dense, Activation
 
model = Sequential()
model.add(Dense(16, input_shape=(64,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=1)

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

