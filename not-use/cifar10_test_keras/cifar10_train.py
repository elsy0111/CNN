# チュートリアルのコード(日本語説明付き)

# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
 
"""A binary to train CIFAR-10 using a single GPU.
 
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
 
Speed: With batch_size 128.
 
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
 
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
 
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
from datetime import datetime
import os.path
import time
 
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
#import cifar-10-python.tar.gz
 
#from tensorflow.models.image.cifar10 import cifar10
import cifar10
 
FLAGS = tf.app.flags.FLAGS
 
# 訓練済データの格納先ディレクトリ (自動的に作成するので、事前にディレクトリを作成する必要はありません)
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
 
# 訓練回数 (100万回⇒1万回に変更)
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
 
# True に設定すると、計算用に割り当てられているデバイスをログ出力する
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
 
 
#
# 訓練を行う関数
#
def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
 
        # CIFAR-10.の画像データとラベルを取得
        images, labels = cifar10.distorted_inputs()
 
        # 予測モデルを計算するためのグラフを作成
        logits = cifar10.inference(images)
 
        # ロス値を計算
        loss = cifar10.loss(logits, labels)
 
        # 1回ごとのバッチサンプルを利用して学習し、モデルのパラメータを更新
        train_op = cifar10.train(loss, global_step)
 
        # Saver (学習途中のデータを保存する機能) を作成
        saver = tf.train.Saver(tf.all_variables())
 
        # 処理のサマリを作成
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
 
        # 全ての変数を初期化
        init = tf.initialize_all_variables()
 
        # セッションを開始し、初期化を実行
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
 
        # Queue Runner (キューによる実行) を開始
        tf.train.start_queue_runners(sess=sess)
 
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
 
        # 設定した学習回数分、繰り返し実行
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
 
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
 
            # 10 回ごとにロスと学習速度を表示
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
 
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
 
            # 100 回ごとにサマリを出力
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
 
            # 定期的(1,000回毎、または最大学習回数に達した際) 学習したモデルを保存
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
 
 
def main(argv=None):  # pylint: disable=unused-argument
    # データセットをダウンロードし、解凍
#    cifar10.maybe_download_and_extract()
    # 訓練済データが存在する場合、削除
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
  
    # 訓練済データ格納先フォルダを作成
    tf.gfile.MakeDirs(FLAGS.train_dir)
 
    # 訓練を実行
    train()
 
 
 
if __name__ == '__main__':
    tf.app.run()