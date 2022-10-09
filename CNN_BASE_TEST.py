# 機械学習(cnn)を行うプログラム
# 学習用データはDatasetのファイル内にある make-dataset.py にて事前に作成する必要があります
# 実験用です
# (predictionの出力はused_rangeを変更すれば好きなとこに変更できます)
# [dataset置き場]https://xs011755.xsrv.jp/Files/PROCON/All_Dataset/03/
# [正規化]https://www.tensorflow.org/addons/tutorials/layers_normalizations
# [separable conv2d]https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/SeparableConv2D
# [inception]https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
# [DenseNet]https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet

#*----- import -----
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import os
#*----- import -----
F = False
T = True

#*----- 初期値設定 -----
EPOCHS = 0  # 学習回数
used_range = range(0*2, 44*2)  # 使用した読みのデータの範囲
test_size = 0.16               # テストデータの割合

dataset_count = 8
kernel_shape = (4,7)

#*----- 読み込み -----
if dataset_count == 1:
    load_model = F
    os.remove("saved_model/my_model_88in5.h5")
    empty_file = open("saved_model/my_model_88in5.h5", 'w')
    empty_file.close()
else:
    load_model = T
print("load_model: ",load_model)
print("dataset_count: ",dataset_count)
load_data = 'saved_model/my_model_88in5.h5'          # 読み込むh5ファイルを指定  
save_data = load_data # 保存するファイル(load_dataにすると上書き保存)
#Dataset = 'Dataset/Dataset_4003_1-44_10'  # 使用するデータセット
Dataset = "../DATASET/88in5/Dataset_3000_88in5(" + str(dataset_count) + ")"
Dataset = "../DATASET"
#*--------------------------------------------------------

N = 5

# 合成数
show_summary = F # model.summary() を実行する場合は True
save = T         # model.save(my_model) を実行する場合は True

view_predictions = 10 # 予測値表示数
batch_size = 2**3     # バッチサイズ

#*----- compile用 -----
metrics = [tf.keras.metrics.Precision()]  # thresholds=0 は sigmoid 以外を使用する際に設定、闘値のこと
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)  #クロスエントロピー誤差
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000008)#!

#* -----画像サイズ -----
high = int(250)
width = int(250)
list_num = 88

#*----- データセットの読み込み -----
images = np.load(Dataset + '/images.npy')
labels = np.load(Dataset + '/labels.npy')
images = (images + 80)/80
print("len(images)", len(images))        # データセットの数

#*----- データセットを分ける -----
# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = test_size)
test_images = images
test_labels = labels

# print("train_images(len): ", len(train_images)) 
# print("test_images(len): ", len(test_images)) 

print("\n", "学習開始")
#*---------- 学習部分 ----------
def create_model():
    #* 画像処理(特徴を出す)
    rate = 0.30#!
    activation1='swish'
    activation2='tanh'
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(high, width, 1)))
    
    model.add(layers.Conv2D(32, kernel_shape,
                            #activation=activation1,
                            ))

    model.add(tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform"))
    model.add(layers.MaxPooling2D((2, 2)) )
 
    #*----- ニューラルネットワーク -----
    model.add(layers.Flatten())
    # ドロップアウト
    model.add(layers.Dropout(rate))
    model.add(layers.Dense(list_num, activation='sigmoid'))  # 0~1での出力(確率が高いほど1に近づく)
    
    #*----- compile -----
    # 使用するものは初期値設定で指定してます
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics
                  )

    return model

#*----- modelの読み込み -----
if load_model == True:
    model = tf.keras.models.load_model(load_data)
else:
    model = create_model()

if show_summary == True:
    model.summary()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)

print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

if load_model == False:
    print("use  create_model")
#*----- 学習開始 -----
#** 学習率の減衰自動化
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_precision', # 監視対象(val_loss or val_precision)
    factor = 0.8, # 減少する割合。new_lr = factor * lr
    patience = 3, # 何エポック変化が見られなかったら変更するか
    verbose = 1,  # 学習率減少時にメッセージを表示
    mode = 'max', # 監視する値の増加が停止した際に変更(min, auto も選択可能)
    epsion = 0.060, # 改善があったと判断する閾値
    cooldown = 0,
    min_lr = 0.000003) # 減少する限度

"""
# 学習部分
history = model.fit(train_images, train_labels, batch_size=batch_size,
                    epochs=EPOCHS, 
                    validation_data=(test_images, test_labels),
                    callbacks = [reduce_lr]                  
                    )

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# データの保存
if save == True:
    model.save(save_data)

"""

#*----- テスト結果の出力 -----
test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)
print(" ")
print("test_accuracy: ", test_accuracy)
print("test_loss:     ", test_loss)

#* データ数の分だけ予測値などのインデックスを出力する関数
def make_answerlist(answer_list,n):
    result = []
    decoy = np.array(answer_list)
    result = decoy.argsort()[::-1]
    result += 1
    idx_88 = []
    for i in range(n):
        idx_88.append(result[i])
    # print("idx_88:       ", idx_88)
    idx_44 = []
    #idx_88 = sorted(idx_88)
    for idx in idx_88:
        if idx % 2 == 0:
            idx = int(idx/2)
            idx_44.append(idx)
        else:
            idx += 1
            idx = int(idx/2)
            idx_44.append(idx)
    #print("idx_44:       ", idx_44)
    idx_44 = sorted(idx_44)
    return idx_44

#*----- 予測値・正解地の出力 -----
# # prediction: 予測値
prediction = model.predict(test_images)
n = 0   # カウンター用
same_len_set = []
for j in prediction:
    i = test_labels[n]
    pre_44 = make_answerlist(j, N)
    acc_44 = make_answerlist(i, N)
    if n < view_predictions:
        print("\n[予測値 : 正解値]", end = "")
        cnt = 1
        for k in used_range:
            if k % 8 == 0:
                print("")
            pre_format = format(j[k], '.5f')
            if cnt <= 9:
                print(cnt,"",pre_format, end = " : ")
                print(i[k],end = "   ")
            else:
                print(cnt,pre_format, end = " : ")
                print(i[k],end = "   ")
            cnt += 1
        print("\n[予測値]")
        print(pre_44)
        print("[正解値]")
        print(acc_44)
        print("予測: ", pre_44)
        print("正解: ", acc_44)

    n += 1
    pre_44_set = set(pre_44)
    acc_44_set = set(acc_44)
    same_set = pre_44_set & acc_44_set
    same_len_set.append(len(same_set))
    if n < view_predictions:
        print("正解数: ",len(same_set))


print("")
print("テスト数↓: ",len(same_len_set))
print("正解数 平均: ",np.average(same_len_set))
print("\ntest_accuracy: ", test_accuracy)
print("test_loss: ", test_loss)
print("images: ", len(images))

print("画像総数", len(images))
print("test_images: ", len(test_images))
