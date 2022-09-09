# https://www.tensorflow.org/tutorials/load_data/images?hl=ja
# [課題]データセットを使用したため、ラベルと画像をそれぞれの変数に分ける方法が不明
# ------>学習部分が少し違う

#---------- import ----------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib
#---------- datesetの読み込み ----------
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)
# 画像の枚数の出力
image_count = len(list(data_dir.glob('*/*.jpg')))
print("画像の数 : ", image_count)

#roses = list(data_dir.glob('roses/*'))
#PIL.Image.open(str(roses[0]))

# パラメーターを定義する
batch_size = 32  # バッチサイズ
img_height = 180 # 縦のピクセル数
img_width = 180  # 横のピクセル数

# 画像の8割をトレーニング用に、2割を検証用にする
# train_ds: 画像, ラベルの入ったトレーニング用データセット
# val_ds  : 同じく、検証用データセット
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# Found 3670 files belonging to 5 classes.
# Using 2936 files for training.

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# Found 3670 files belonging to 5 classes.
# Using 734 files for validation.

class_names = train_ds.class_names
print(class_names) 
# ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

#[↓↓↓出力されません笑]
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): ##
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
# 画像のバッチ確認
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)  # (32, 180, 180, 3)--->(枚数, 縦, 横, 高さ(RGB or Gray))
  #print(type(image_batch))
  print(labels_batch.shape) # (32,)--->(枚数, )
  break
# image_batch は形状(32, 180, 180, 3)のテンソル。
# 形状180*180*3の32枚の画像のバッチ
# labal_batch は形状(32, )のテンソル。
# 32枚の画像に対応するラベル

# データの正規化
# 画像のバッチの値を再スケーリングおよびオフセットします
# (たとえば、[0, 255] 範囲の入力から [0, 1] 範囲の入力に移動します。)
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# ピクセル数が `[0,1]`になっていることに注意してください
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print(type(train_ds))
#exit()
#---------- model ----------
num_classes = 5
model = tf.keras.Sequential([
  # 入力値を新しい範囲に再スケーリングする前処理レイヤー
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
