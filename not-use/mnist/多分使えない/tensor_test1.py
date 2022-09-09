# coding:utf-8
#TensorFlowをインポート
#TensorFlowのバックエンドではC++の高速なライブラリを使用しています
import tensorflow as tf
#MNISTデータのロード
from mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import time
import os,sys
import numpy as np
from PIL import Image
from matplotlib import pylab as plt
start = time.time()
tf.reset_default_graph()
sess = tf.InteractiveSession()
#プレースホルダーにてTFに計算を依頼
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
###############多層畳み込みネットワークの構築####################
#重みの初期化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
##畳み込み
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
##プーリング(次元削減)
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#######畳み込み層第1層############################
W_conv1 = weight_variable( [5, 5, 1, 32] )
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#######畳み込み層第2層############################
W_conv2 = weight_variable( [5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
########密集接続層######################
W_fc1 = weight_variable( [7*7*64, 1024] )
b_fc1 = bias_variable( [1024] )
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64] )
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#########読みだし層#################
W_fc2 = weight_variable( [1024, 10] )
b_fc2 = bias_variable( [10] )
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#####訓練######
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predition = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, "float"))
#初期化
sess.run(tf.initialize_all_variables())
#学習データ保存用
saver = tf.train.Saver()
#保存用ディレクトリの作成
os.makedirs("./mnist_model_2", exist_ok=True)

ckpt = tf.train.get_checkpoint_state('./mnist_model_2')
if ckpt: # 学習データがある場合
    print("学習データを読み込みます")
    print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path) # データの読み込み
else: #学習データがない場合→学習を開始
    print("学習を開始します")
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 ==0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
            print("学習回数： %d 回, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    ##学習パラメータの保存
    saver.save(sess, "./mnist_model_2/CNN.ckpt")
print("この学習の正解率は、")
#学習結果の表示
print("この学習の正解率は %g です"%accuracy.eval(feed_dict={ x: mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))
#ここから手書きの文字を認識させる
#ファイルを開く
data_dir_path = u"./numbers/"
file_list = os.listdir(r'./numbers/')
for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    if ext == u'.png' or u'.jpeg' or u'.jpg':
        abs_name = data_dir_path + '/' + file_name
        #グレースケールとして画像の読み込み
        img = Image.open(abs_name).convert('L')
        # バックを白に
        img = Image.frombytes('L', img.size,
                                bytes(map(lambda x: 255 if (x > 160) else x,
                                          img.getdata())))
        print (abs_name+"の画像")
        plt.imshow(img)
        plt.pause(1)
        img.thumbnail((28, 28))
        # input_data の形に揃える
        img = map(lambda x: 255 - x, img.getdata())
        img = np.fromiter(img, dtype=np.uint8)
        img = img.reshape(1, 784)
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)
        #学習データと読み込んだ数値との比較を行う
        p = sess.run(y_conv, feed_dict={x:img, y_: [[0.0] * 10], keep_prob: 0.5})[0]
        #最も可能性のある数字を表示
        print (abs_name+"の認識結果は.....................................")
        print ("*******************************")
        print (np.argmax(p))
        print ("*******************************")
sess.close()
timer = time.time() - start
print("time:{0}".format(timer) + "[sec]")