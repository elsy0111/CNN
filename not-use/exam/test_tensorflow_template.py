# https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?hl=ja#6
# tensorflowによる手書き数字の認識
# データセット⇒MNIST
# nextTrainBatch(batchSize): トレーニング セットから画像とそのラベルをランダムに返します。
# nextTestBatch(batchSize): テストセットから画像とそのラベルのバッチを返します。
# 

# TensorFlow を読み込み
import tensorflow as tf
 
# 入力データ格納用の 784 px 分のプレースホルダを作成
x = tf.placeholder(tf.float32, [None, 784])
 
# 重み (784 x 10 の行列) の Variable を定義
W = tf.Variable(tf.zeros([784, 10]))
 
# バイアス (長さ 10 の行列) の Variable を定義
b = tf.Variable(tf.zeros([10]))
 
# ソフトマックス回帰による予測式を定義
y = tf.nn.softmax(tf.matmul(x, W) + b)
 
# 出力データ (予測値) 格納用のプレースホルダ
y_ = tf.placeholder(tf.float32, [None, 10])
 
# 交差エントロピーを最小化するよう、学習を行う式を以下のように定義
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
 
# 初期化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
 
# 学習を 1,000 回繰り返す
for i in range(1000):
  # 訓練用データから 100 件取得
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # train_step を実行
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})