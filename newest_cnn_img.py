
# 実験用です
# (predictionの出力はused_rangeを変更すれば好きなとこに変更できます)
# [dataset置き場]https://xs011755.xsrv.jp/Files/PROCON/All_Dataset/03/
# [正規化]https://www.tensorflow.org/addons/tutorials/layers_normalizations
# [separable conv2d]https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/SeparableConv2D
# [inception]https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
# [DenseNet]https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet
#*----- import -----
#import os
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
#*----- 初期値設定 -----
EPOCHS = 10  # 学習回数
used_range = range((1-1)*2, 22*2+1)   # 使用した読みのデータの範囲(+1は例外も１つ出力するため)
#*----- 読み込み -----
load_data = 'saved_model/' + 'my_model-07_test-28.h5'    # 読み込むh5ファイルを指定  
save_data = load_data # 保存するファイル(load_dataにすると上書き保存)

Dataset = '../DATASET' + '/Dataset_2000_44in5'
images = np.load(Dataset + '/images.npy')
labels = np.load(Dataset + '/labels.npy')
#*-----------------------------------------------------------
N = 5   # 合成数
view_predictions = 25

batch_size = 2**3
F = False
T = True
load_model = F  #! H5ファイルを読み込む場合は True
show_summary = F # model.summary() を実行する場合は True
save = F        # model.save(my_model) を実行する場合は True
#* metricsの指定
metrics=[tf.keras.metrics.Precision()]  # thresholds=0 は sigmoid 以外を使用する際に設定、闘値のこと
#* lossの指定 
loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)  #クロスエントロピー誤差
#* optimizarの指定
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1 = 0.9, 
    beta_2 = 0.999, 
    epsilon = 1e-07,
    amsgrad = False,
    name = 'Adam')
#optimizer = 'adam'
#* -----画像サイズ -----
high = int(250)
width = int(250)
list_num = 88

#*----- データセットの読み込み -----

images = (images + 80)/80
print("len(images)", len(images))        # データセットの数

#*----- データセットを分ける -----
train_images = []
train_labels = []
test_images = []
test_labels = []
# テストデータの割合を指定
test_size = 0.20
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
    
    model.add(layers.Conv2D(32, (5, 5),
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
    #model.add(layers.LayerNormalization(axis=3 , center=True , scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
 
    #*----- ニューラルネットワーク -----
    model.add(layers.Flatten())
    # ドロップアウト
    rate = 0.2
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
    n = 0   # カウンター用
    for j in prediction:
        i = test_labels[n]
        print("\n[予測値 : 正解値]", end = "")
        for k in used_range:
            if k % 6 == 0:
                print("")
            pre_format = format(j[k], '.5f')
            print(pre_format, end = " : ")
            #print("")
            print(i[k], end = "      ")
            
        n += 1
        print("")
        if n == view_predictions:
            break
        
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
print("train_images: ", len(train_images))
#print(prediction[n-1])
exit()
#*----- グラフ用 -----
def plot_loss_accuracy_graph(history):
    # 青い線で誤差の履歴をぷろっと、検証時誤差は色の薄い線
    plt.plot(history.history(metrics), "-o", color = (0, 0, 1), label = 'train_loss', linewidth = 2)
    plt.plot(history.history['val_loss'], "-o", color = (0, 0, 1, 0.35), label = 'val_loss', linewidth = 1.5)
    plt.title('LOSS')
    plt.xlabel('Epochs')
    
    # 緑の線で精度の履歴をぷろっと、検証時精度は色の薄い線
    plt.plot(history.history(metrics), "-o", color = (1, 0, 0), label = 'train_accuracy', linewidth = 2)
    plt.plot(history.history['val_accuracy'], "-o", color = (1, 0, 0, 0.35), label = 'val_accuracy', linewidth = 1.5)
    plt.title('LOSS & ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy & loss')
    plt.legend(loc="upper right")  # 凡例の場所（右上）
    plt.show()
  
# グラフの表示(loss, accuracy, epochs)  
plot_loss_accuracy_graph(history)
exit()
#*----------------------------------------------------------------
# 以下、saved_modelを使用した保存データを読み込むやつ(使用しない)
#* 保存したモデルから新しい Keras モデルを再度読み込みます。
new_model = tf.keras.models.load_model('saved_model/my_model')

# そのアーキテクチャを確認する
new_model.summary()

# 復元されたモデルを評価する
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

# HDF5ファイルとして
# Keras は HDF5 の標準に従ったベーシックな保存形式も提供します。
# 新しいモデルインスタンスを作成し、学習させる。
model = create_model()
EPOCHS = EPOCHS 
model.fit(train_images, train_labels, epochs=EPOCHS)

# モデル全体を HDF5 ファイルに保存する。
# 拡張子 '.h5' は、モデルが HDF5 に保存されることを示す。
model.save('my_model.h5')

# 保存したファイルを使ってモデルを再作成します。
# 重みとオプティマイザを含め、全く同じモデルを再作成する。
new_model = tf.keras.models.load_model('saved_model/my_model')

# そのアーキテクチャを確認する
# モデルのアーキテクチャを表示する
new_model.summary()
# 正解率を検査します
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))