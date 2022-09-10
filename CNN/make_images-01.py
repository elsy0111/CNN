# Merged_all.py + CNN
# データを作成して、検証するプログラム

#-----IMPORT-----#
from itertools import chain
import os
import datetime
import shutil
from random import randint
import numpy as np
import librosa
from scipy.io.wavfile import read, write
#-----IMPORT-----#

#-----IMPORT-----#
import librosa.display
import imageio
#-----IMPORT-----#

#*----- import -----
from operator import imod
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib

import glob
import cv2
from sklearn.model_selection import train_test_split
#*----- import -----

PCM = 48000

#--------------Set Parameter--------------#
fft_size = 2048                 # Frame length
hl = int(fft_size / 4)          # Frame shift length
hi = 250                        # Height of image
wi = 250 - 1                    # Width of image
F_max = 20000                   # Freq max
window = np.blackman(fft_size)  # Window Function
#--------------Set Parameter--------------#
#* 初期値
count = 0
dataset_num = 10  # 繰り返す回数
#* cnn で使用(格納用)
images = []  # 画像格納用
labels = []  # ラベル格納用
#*---------------------------------------------------
for count in range(dataset_num):
    ValueErr = 0
    #* 以下の数行は削除予定(そのままcnnに組み込むため)
    dt_now = datetime.datetime.now()

    Dataset_dilectory_name = dt_now.strftime('%m%d_%H%M%S%f')
    Dataset_dilectory_name = "audio/Conposition_Audio/" + Dataset_dilectory_name
    os.mkdir(Dataset_dilectory_name)

    f = open(Dataset_dilectory_name + '/meta_data.txt', 'w')

#--------------Make Random List(length = 88)--------------#
    #* ランダムな数を作成
    N = randint(3,20)     #! No DEBUG

    t = []

    while len(t) < N:
        j = randint(1,44)
        t.append(j)
        t = list(set(t))
    t.sort()

    s_list = [0] * 44

    for i in t:
        s_list[i-1] = 1

    cnt = 0
    list88 = [0] * 88
    for i in s_list:
        if i == 1:
            j = randint(0,1)
            if j == 1:
                list88[cnt] = 1
                list88[cnt + 1] = 0
            else:
                list88[cnt] = 0
                list88[cnt + 1] = 1
        cnt += 2

    print("answer_label : ",list88)
    
#--------------Make Random List(length = 88)--------------#


#? out meta_data
    f.write("合成データ数" + "\n" + str(N) + "\n")
    f.write("正解ラベル" + "\n" + str(list88) + "\n")


#--------------Make filename by list88--------------#
    n_audio = 0
    audio_list = []


    for i,j in enumerate(list88):
        if j == 1:
            if i%2 == 0: #日本語
                i = int(i/2) + 1
                if len(str(i)) == 1:
                    l = "J0" + str(i)
                else:
                    l = "J" + str(i)
            else:#英語
                i = int(i/2) + 1
                if len(str(i)) == 1:
                    l = "E0" + str(i)
                else:
                    l = "E" + str(i)
            audio_list.append(l)
            n_audio += 1


#? out meta_data
    f.write("合成元(種類)" + "\n" + str(audio_list) + "\n")
#--------------Make filename by list88--------------#



#--------------Make delay_list--------------#
    all_data = []
    delay_list = []
    raw_audio_length_list = []

    for i,name in enumerate(audio_list):
        PCM, data = read("audio/Sample_Audio/"+name+".wav")
        raw_audio_length_list.append(len(data))
        delay_random_num = randint(0, 5) * 4800    #! random delay No DEBUG
        delay_list.append(delay_random_num)
        cut_offset_data = data[delay_random_num:]
        all_data.append(cut_offset_data)
        
    audio_length_list = []

    for data in all_data:
        audio_length_list.append(len(data))


    raw_audio_length_list = np.array(raw_audio_length_list)
    delay_list = np.array(delay_list)
#--------------Make delay_list--------------#

#? out meta_data
    f.write("Delay" + "\n" + str(delay_list) + "\n")


#------------------Fill Zero----------------#
    max_audio_length = max(audio_length_list)

    result = np.zeros(max_audio_length,dtype = int)

    for data in all_data:
        n_empty = max_audio_length - len(data)
        empty_list = np.zeros(n_empty,dtype = int)
        long_data = list(chain(data,empty_list))
        result += long_data

#------------------Fill Zero----------------#


#------------------Delete------------------#
    TorF = True

    while TorF:
        cnt = 0
        n_split = randint(2,5) #! n_split
        while (cnt < 100):
            cnt += 1
            delete_num = randint(0,250000)
            if delete_num <= len(result) - 0.5 * 48000 * n_split:
                if (len(result) - delete_num)/n_split <= 48000 * 3:
                    TorF = False
                    break   # ok

    result = result[:len(result) - delete_num]
#------------------Delete------------------#

#? out meta_data
    f.write("冒頭,末尾削除" + "\n" + str(delete_num) + "\n")

#------------------Export audio----------------#
    result = np.array(result,dtype = float)
    result /= 2**15

    writefile = Dataset_dilectory_name +  "/out.wav"
    write(writefile,rate = PCM,data = result)
#------------------Export audio----------------#

    wav_file_name = writefile

    data,PCM = librosa.load(wav_file_name,sr = PCM)

    frames = len(data)
    sec = frames/PCM

#-----------------cut list------------------
    c = True
    while c:
        split_list = []
        for i in range(n_split - 1):
            split_list.append(randint(1,frames))
        split_list.sort()
        split_list.insert(0,0)
        split_list.append(frames)
        c = False
        for i in range(n_split):
            if split_list[i + 1] - split_list[i] <= 0.5 * 48000:
                c = True
#-----------------cut list------------------

#-----------------cut audio------------------
#? out meta_data
    f.write("分割" + "\n" + str(split_list) + '\n')

    split_list[-1] += 1

    os.mkdir(Dataset_dilectory_name + "/split")

    for j in range(n_split):
        split_data = data[split_list[j]:split_list[j + 1]]
        n_empty = 48000 * 3 - len(split_data)
        try:
            empty_list = np.zeros(n_empty)
        except ValueError:
            print("value Error (split_data is too large)")
            f.close()
            shutil.rmtree(Dataset_dilectory_name)
            ValueErr = 1
            break
        same_length_data = np.array(list(chain(split_data,empty_list)))
        out = Dataset_dilectory_name + '/split/out_' + str(j + 1) + '.wav'
        write(out,rate = PCM,data = same_length_data)
#---------------------------Make Audio end-----------------------------#

    if ValueErr == 1:
        continue
    
    os.mkdir(Dataset_dilectory_name + '/images')

    for j in range(n_split):
        out = Dataset_dilectory_name + '/split/out_' + str(j + 1) + '.wav'
        
#--------------Load Audio File--------------#
        wav_file_name = out

        data, PCM = librosa.load(wav_file_name,sr = PCM)
#--------------Load Audio File--------------#

        data = data[0:wi*hl]

#--------------STFT--------------#
        S = librosa.feature.melspectrogram(
            y = data, sr = PCM, n_mels = hi, fmax = F_max, hop_length = hl, 
            win_length = fft_size, n_fft = fft_size, window = window)

        S_dB = librosa.power_to_db(S, ref = np.max)
#--------------STFT--------------#

# S_dB.sort(reverse=True)
        S_dB = np.flipud(S_dB)
        imageio.imwrite(Dataset_dilectory_name + "/images/" + str(j + 1) + '.png', S_dB)
        
        #* データセット作成
        img_data = imageio.imread(Dataset_dilectory_name + "/images/" + str(j + 1) + '.png')
        """img = glob.glob(Dataset_dilectory_name + "/images/" + str(j + 1) + '.png')
        img_data = cv2.imread(img)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)"""
        labels.append(np.array(list88))    #* ラベルに追加
        images.append(img_data)  #* 画像に追加 
                
    f.close()
    
#print(images)
#print(labels)
#*------------------------------------------------------------------------

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