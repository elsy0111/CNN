#*----- import -----
from operator import imod
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
#*----- import -----

images = np.load("Dataset_1000/images.npy")

labels = np.load("Dataset_1000/labels.npy")

#*----- ここからCNN -----

#*----- ここからCNN -----
high = int(250)
width = int(250)

# データセットを分ける
train_images = []
train_labels = []
test_images = []
test_labels = []

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.1)

# 訓練用データ、テストデータに取り込んだデータを格納する
train_images = (np.array(train_images) + 80) / 160
train_labels =  np.array(train_labels)
test_images  = (np.array(test_images)  + 80) / 160
test_labels  =  np.array(test_labels)

# 画像サイズなどの確認用
print("train_images.shape: ", train_images.shape) # (枚数, 縦, 横)
print("train_labels.shape: ", train_labels.shape) # (枚数,)
print("test_images.shape: ", test_images.shape) # (枚数, 縦, 横)
print("test_labels.shape: ", test_labels.shape) # (枚数,)
'''print("train_images[0]: ", train_images[0]) # ndarray(画像の要素)
print("train_labels[0]: ", train_labels[0]) # ndarray(ラベルの要素)
'''

#---------- 学習部分 ----------
# 画像処理(特徴を見つける)
# 活性化関数: relu
# layers: Conc2D(畳み込み層), MaxPooling2D(プーリング層)

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', 
          input_shape=(high, width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.summary()

#----- ニューラルネットワーク -----
# layers.Flatten: 数値を1次元にしてる(画像を線にするイメージ?)
# 活性化関数: relu
# Dense: 全結合層
# 数値の収束に'softmax'を使用してる(省略してます)

model.add(layers.Flatten())

rate = 0.6
model.add(layers.Dropout(rate))
# model.add(layers.SpatialDropout1D(rate))

model.add(layers.Dense(88, activation='sigmoid'))  # 0~1での出力(確率が高いほど1に近づく)
# パラメーターなどの表示(省略可)
model.summary()

# [optimizerの種類]https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#* lossの指定
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # 確率を使用したloss
# loss=tf.keras.losses.MeanSquaredError()  # MSE

#----- compile -----
model.compile(optimizer='adam',
              #optimizer='SGD', 
              loss=loss,
              metrics=['accuracy']
              )


#----- 学習開始 -----
EPOCHS = 50   # 学習回数の指定
history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels))

#----- 機械の予測の出力 -----

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

True_cnt = 0
False_cnt = 0

for j in range(len(prediction)):
    pr = prediction[j]
    ans = test_labels[j]
    pr_m = np.argmax(pr) + 1    # max_index
    ans_m = np.argmax(ans) + 1  # answer_index
    
    # print("----------------------------")
    # print("prediction[j]  : ", pr)
    # print("test_labels[j] : ", ans)
    
    if pr_m == ans_m:
        # print("／＼" , " -index : ", pr_m)
        # print("＼／" , " answer : ", ans_m)
        True_cnt += 1
    else:
        # print("＼／" , " -index : ", pr_m)
        # print("／＼" , " answer : ", ans_m)
        False_cnt += 1

True_Test_Acc = True_cnt/len(prediction)
print("True_cnt  : ", True_cnt)
print("False_cnt : ", False_cnt)
print("Accuracy  : ", True_Test_Acc)
# exit()

import matplotlib.pyplot as plt

#*----- グラフ用(conpile後に置く) -----
def plot_loss_accuracy_graph(history):
    plt.title('True_Test_Acc : ' + str(True_Test_Acc))
    plt.xlabel('Epochs')
    plt.ylabel('Loss & Accuracy')

    plt.ylim([0, 1])
    plt.xlim([0,EPOCHS])

    plt.plot(history.history['loss'], "-o", color = (0, 0, 1), label = 'train_loss', linewidth = 2)
    plt.plot(history.history['accuracy'], "-o", color = (1, 0, 0), label = 'train_accuracy', linewidth = 2)

    x = np.arange(0,EPOCHS)
    y = x * 0 + True_Test_Acc
    plt.plot(x, y, color = (0, 1, 0), label = 'True_Test_Acc', linewidth = 2)

    plt.legend(loc = "lower right")
    plt.yticks(np.arange(0, 1.01, 0.1))

    plt.grid(axis = 'y')
    
plot_loss_accuracy_graph(history)

plt.show()