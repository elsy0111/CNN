# tensorflow での例(mnistを使用)、batchの部分でエラーが出る。(mnistのinputの仕方が違うから？？)
# tensorflow_ver1なのでいくつか修正した。
# [参考]https://qiita.com/SwitchBlade/items/6677c283b2402d060cd0
# [解説付き]https://qiita.com/KojiOhki/items/64a2ee54214b01a411c7
# 少し古い文献なので注意
print('mnist_train.py START')

import imp
import tensorflow as tf
import keras
#from tensorflow.examples.tutorials.mnist import input_data
from keras import backend
from keras.datasets import mnist
import datetime
import numpy as np
import DeepConvNet as CNN

IMAGE_SIZE  = 28    # 画像サイズ
NUM_CLASSES = 10    # 識別数

print('MNIST Download Start')
# MNISTデータのダウンロード
# x_ -- 画像データ, y_ -- ラベル(0~9)
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.array(x_train)/255.0
y_train = np.array(y_train)
# ラベルをone-hot形式へ変更
y_train = keras.utils.to_categorical(y_train, 10)
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True) # ならない
print('MNIST Download End')

""" 損失関数
    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
      cross_entropy: 交差エントロピーのtensor, float
"""
def loss(logits, labels):
    cross_entropy = -tf.compat.v1.reduce_sum(labels*tf.compat.v1.log(logits))
    return cross_entropy

""" 訓練のopを定義する関数
    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数
    返り値:
      train_step: 訓練のop
"""
def training(loss, learning_rate):
    # 勾配降下法(Adam)
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

"""正解率(accuracy)を計算する関数
    引数:
        logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
        labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
        accuracy: 正解率(float)
"""
def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.compat.v1.argmax(logits, 1), tf.compat.v1.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

if __name__=="__main__":
    with tf.Graph().as_default():
        print('設定 START') 
        x_train = tf.compat.v1.placeholder("float", shape=[None, IMAGE_SIZE * IMAGE_SIZE])  # 入力
        y_train = tf.compat.v1.placeholder("float", shape=[None, NUM_CLASSES]) # 出力
        keep_prob = tf.compat.v1.placeholder("float")    #ドロップアウト

        # モデルを作成
        logits = CNN.CNN.makeMnistCNN(x_train, keep_prob , IMAGE_SIZE , NUM_CLASSES) 

        # opを定義
        loss_value = loss(logits, y_train) 
        train_op = training(loss_value,1e-4) 
        accur = accuracy(logits, y_train) 

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        #TensorBoardで追跡する変数を定義
        accuracy_op_train = tf.compat.v1.summary.scalar("Accuracy on Train", accur)
        accuracy_op_test = tf.compat.v1.summary.scalar("Accuracy on Test", accur)
        summary_op_train = tf.compat.v1.summary.merge([accuracy_op_train])
        summary_op_test = tf.compat.v1.summary.merge([accuracy_op_test])
        summary_writer = tf.compat.v1.summary.FileWriter("./tensorflow_ver1/TensorBoard", graph=sess.graph)

        # 訓練したモデルを保存
        # (tf.train.Saver()が呼ばれる前までに呼ばれた引数が対象になる)
        saver = tf.compat.v1.train.Saver()
        print('設定 END')

        print('学習 START : ' + str(datetime.datetime.now()))
        #学習の実行
        for epoch in range(5000):
            #訓練データセットから 50 のランダムなデータの “バッチ” を取得 [0]に画像の配列、[1]に結果の配列
            # 以降、bacheの部分がうまくいかない(別の方法を探すしかなさそう)
            batch = mnist.train.next_batch(50)

            # 学習の途中経過の表示・TensorBoard書き込み
            if epoch % 100 == 0:
                train_accury = sess.run(accur, feed_dict={x_train: batch[0], y_train: batch[1], keep_prob: 1.0})

                # テストデータ(検証データ)で評価
                test_batch = mnist.validation.next_batch(500, shuffle=False)
                test_accury = sess.run(accur, feed_dict={x_train: test_batch[0], y_train: test_batch[1], keep_prob: 1.0})
                # ↓ Jupiterで実行するとコンソールが落ちる (メモリ不足？)
                #test_accury = sess.run(accur, feed_dict={x_train: mnist.validation.images, y_train: mnist.validation.labels, keep_prob: 1.0})
                print("epoch:%d, train_accury : %g  test_accury : %g"%(epoch, train_accury , test_accury))

                summary_str_train = sess.run(summary_op_train, feed_dict={x_train: batch[0], y_train: batch[1], keep_prob: 1.0})
                summary_writer.add_summary(summary_str_train, epoch)

                summary_str_test = sess.run(summary_op_test, feed_dict={x_train: test_batch[0], y_train: test_batch[1], keep_prob: 1.0})
                #summary_str = sess.run(summary_op_test, feed_dict={x_train: mnist.validation.images, y_train: mnist.validation.labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str_test, epoch)
                summary_writer.flush()

            # 学習
            sess.run(train_op, feed_dict={x_train: batch[0], y_train: batch[1], keep_prob:0.5})

        print('学習 END : ' + str(datetime.datetime.now()))

        #結果表示 (テストデータで評価)
        test_batch = mnist.test.next_batch(500, shuffle=False)
        print("test accuracy : %g" %sess.run(accur, feed_dict={x_train: test_batch[0], y_train: test_batch[1], keep_prob: 1.0}))
        #print("test accuracy : %g" %sess.run(accur, feed_dict={x_train: mnist.test.images, y_train: mnist.test.labels, keep_prob: 1.0}))

        save_path = saver.save(sess, "./ckpt/model.ckpt") # 変数データ保存
        print('Save END : ' + save_path )

        summary_writer.close()
        sess.close()

        print('mnist_train.py END')
        
    #TensorBoardで追跡する変数を定義
    accuracy_op_train = tf.compat.v1.summary.scalar("Accuracy on Train", accur)
    accuracy_op_test = tf.compat.v1.summary.scalar("Accuracy on Test", accur)
    summary_op_train = tf.compat.v1.summary.merge([accuracy_op_train])
    summary_op_test = tf.compat.v1.summary.merge([accuracy_op_test])
    summary_writer = tf.compat.v1.summary.FileWriter("./tensorflow_ver1/TensorBoard", graph=sess.graph)
    summary_str_train = sess.run(summary_op_train, feed_dict={x_train: batch[0], y_train: batch[1], keep_prob: 1.0})
    summary_writer.add_summary(summary_str_train, epoch)

    test_batch = mnist.validation.next_batch(500, shuffle=False)
    summary_str_test = sess.run(summary_op_test, feed_dict={x_train: test_batch[0], y_train: test_batch[1], keep_prob: 1.0})
    summary_writer.add_summary(summary_str_test, epoch)
    summary_writer.flush()