
# 実験用です
# (predictionの出力はused_rangeを変更すれば好きなとこに変更できます)
# [正規化]https://www.tensorflow.org/addons/tutorials/layers_normalizations
# [separable conv2d]https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/SeparableConv2D
# [inception]https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
# [DenseNet]https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet
#*----- import -----
#import os
from unicodedata import name
from tensorflow import keras
import numpy as np
from scipy.io.wavfile import read, write
from operator import imod
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow_addons as tfa
#*----- import -----

F = False
T = True

#*----- 初期値設定 -----
load_data = 'my_model-03_test.h5' # 読み込むh5ファイルを指定  
save_data = load_data # 保存するファイル(load_dataにすると上書き保存)

dataset_img = "../DATASET/Dataset_700_18in7/images_reshape.npy"
dataset_lab = "../DATASET/Dataset_700_18in7/labels.npy"

N = 7   # 合成数
used_range = range(0, 9*2+1)   # 使用した読みのデータの範囲(+1は例外も１つ出力するため)

EPOCHS = 30  # 学習回数
batch_size = 2**4

load_model = F  #! H5ファイルを読み込む場合は True
show_summary = T # model.summary() を実行する場合は True
save = T        # model.save(my_model) を実行する場合は True

#* metricsの指定
#metrics = ['Accuracy']
# Precisionは精度は以前とあまり変わらないが、時間がかかる
metrics=[tf.keras.metrics.Precision()]  # thresholds=0 は sigmoid 以外を使用する際に設定、闘値のこと

#* lossの指定 
loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)  #クロスエントロピー誤差

#* optimizarの指定
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)

#* -----画像サイズ -----
high = int(1) #! input_shape
width = int(144000/2)
list_num = 88

#*----- データセットの読み込み -----
images = np.load(dataset_img)
labels = np.load(dataset_lab)

print("Dataset num : ", len(images))        # データセットの数

#*----- データセットを分ける -----
train_images = []
train_labels = []
test_images = []
test_labels = []
# テストデータの割合を指定
test_size = 0.12
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = test_size)

print("train_images.shape: ", train_images.shape) # (枚数, 縦, 横)
print("test_images.shape: ", test_images.shape) # (枚数, 縦, 横)

print("\n", "学習開始")
#*---------- 学習部分 ----------
def create_model():
    #* 画像処理(特徴を出す)
    # 活性化関数: activation
    # layers: Conv2D(畳み込み層), MaxPooling2D(プーリング層)
    activation1='swish'
    activation2='tanh'
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(high, width, 1)))
    
    model.add(layers.Conv2D(32, (1, 5),
                            #groups=1, # グループを2つに分ける?
                            #padding = "same",
                            activation=activation2,
                            ))
    #*xception---まだ書いてない
    #? インスタンス正規化
    model.add(tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform"))
    # model.add(layers.LayerNormalization(axis=3 , center=True , scale=True))
    # model.add(layers.MaxPooling2D((1, 2)) )
    # model.add(layers.Conv2D(32, (1, 3), activation=activation1))
    model.add(layers.MaxPooling2D((1, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation=activation1))
    #model.add(layers.MaxPooling2D((2, 2)))

    #*----- ニューラルネットワーク -----
    # layers.Flatten: 数値を1次元にしてる(画像を線にするイメージ?)
    # 活性化関数: relu（なし）
    # Dense: 全結合層
    # 数値の収束に'sigmoid'を使用してる
    model.add(layers.Flatten())
    # ドロップアウト
    rate = 0.16
    model.add(layers.Dropout(rate))
    #? l2 正規化: kernel_regularizer=tf.keras.regularizers.l2(0.001)
    model.add(layers.Dense(list_num, activation='sigmoid'))  # 0~1での出力(確率が高いほど1に近づく)
    
    #*----- compile -----
    # 使用するものは初期値設定で指定してます
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics
                  )

    return model

#!----- modelの読み込み -----
# Recreate the exact same model, including its weights and the optimizer
if load_model == True:
    model = tf.keras.models.load_model(load_data)
else:
    model = create_model()

if show_summary == True:
    # Show the model architecture
    model.summary()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
if load_model == False:
    print("use  create_model")
#*----- 学習開始 -----
#EPOCHS = 13　# ここでepochを再指定
history = model.fit(train_images, train_labels, 
                    epochs=EPOCHS, batch_size=batch_size,
                    validation_data=(test_images, test_labels)                    
                    )
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
# データの保存
if save == True:
    model.save(save_data)
#*----- テスト結果の出力 -----
test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)
print(" ")
print("test_accuracy: ", test_accuracy)
print("test_loss:     ", test_loss)

#exit()
#? 以下は省略可
#*----- 予測値 -----
if N != 1:
    # prediction: 予測値
    prediction = model.predict(test_images)
    prediction = prediction[:10]
    n = 0   # カウンター用
    for j in prediction:
        i = test_labels[n]
        print("\n[予測値 : 正解値]")
        for k in used_range:
            pre_format = format(j[k], '.5f')
            print(pre_format, end = " : ")
            #print("")
            print(i[k], end = "      ")
            if k % 5 == 0:
                print("")
        n += 1
        print("")
        
else:
    # prediction: 予測値
    prediction = model.predict(test_images)
    n = 0   # カウンター用
    for j in prediction:
        print("予測値\n", j)       
        print("予測値:", max(j))
        print("予測(1~88):", np.argmax(j) + 1)
        i = test_labels[n]
        #print("正解値\n", i)
        print("正解(1~88):", np.argmax(i) + 1)
        #print("")
        n += 1
        
print("\ntest_accuracy: ", test_accuracy)
print("test_loss:     ", test_loss)
#print(prediction[n-1])
exit()