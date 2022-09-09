# 画像の読み込み方法
# [参考]https://youtu.be/ThKRS7B5GFY

from re import X
from tkinter import N
import numpy as np
import tensorflow as tf
import glob
from PIL import Image
# 画像(数値データ)の保存用リスト
x_train = []
y_train = []
n = 1 # カウンター用整数
# ラベルの保存用リスト
#*---------- tensorflow_ver1/Japanese_Allの中の画像を読み込む ----------*#
# tensorflow_ver1/*/*png と指定すれば、ほかのファイルの中の画像を読み込むことが可能
for f in glob.glob("tensorflow_ver1/Japanese_All/*.png"):
    #print(f)
    img_data = tf.io.read_file(f)
    img_data = tf.io.decode_jpeg(img_data)
    #img_data = tf.image.resize(img_data, [25, 50])  # 画像サイズの変更
    
    x_train.append(img_data)
    y_train.append(n)
    n += 1
    #trainとtest, 英語と日本語など分ける場合は以下の指定をする
    
x_train = np.array(x_train) / 255.0
y_train = np.array(y_train)
#print(x_train[1])
#---> 1の羅列にしかならないのは、枠があるからかな...?
#print(y_train)
#今回は１～４４





