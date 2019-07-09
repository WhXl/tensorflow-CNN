# -*- ecoding:utf-8 -*-

import tensorflow as tf


class model():
    def __init__(self):
        self.x_image = tf.placeholder(tf.float32, [None, 784],name='x_image')
        self.y_label = tf.placeholder(tf.float32, [None, 10],name = 'y_label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.x = tf.reshape(self.x_image, [-1, 28, 28, 1])

        # first layer
        # 命名域
        with tf.name_scope('conv1'):
            self.W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name = 'W_conv1')
            # tf.summary.histogram('layer1 weights',self.W_conv1)
            # 创建常数偏移量
            self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name = 'b_conv1')
            # tf.summary.histogram('layer1 biases', self.b_conv1)

            # Convolution
            self.h_conv1 = tf.nn.conv2d(self.x, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv1
            self.h_convRelu1 = tf.nn.relu(self.h_conv1)

        # pooling
        with tf.name_scope('pool1'):
            self.h_pool1 = tf.nn.max_pool(self.h_convRelu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = 'pool1')

        # second layer
        with tf.name_scope('conv2'):
            self.W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),name = 'W_conv2')
            # tf.summary.histogram('layer2 weights', self.W_conv2)
            self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),name = 'b_conv2')
            # tf.summary.histogram('layer2 biases', self.W_conv2)
            # Converlution
            self.h_conv2 = tf.nn.conv2d(self.h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2
            # 激活函数
            self.h_convRelu2 = tf.nn.relu(self.h_conv2)
        # pooling
        with tf.name_scope('pool2'):
            self.h_pool2 = tf.nn.max_pool(self.h_convRelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = 'pool2')
       # 卷积部分最后输出为（高 7 宽 7 通道数 64）
        # 全连接网络
        with tf.name_scope('fc1'):
            ##构建全连接的输入层权重 输入为 7*7*64个节点  隐藏层为 1024个节点
            self.W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name='self.W_fc1')
            tf.summary.histogram('fc1 weights', self.W_fc1)
            ##构建全连接的输入层偏移量
            self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name ='self.b_fc1')
            tf.summary.histogram('fc1 biases', self.b_fc1)
            # 重新设置输入数据的样式
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
            # 进行矩阵乘法
            self.h_fc1 = tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1
            # 激活函数
            self.h_fcRelu1 = tf.nn.relu(self.h_fc1)

        # 防止过拟合
        with tf.name_scope('dropout'):
            # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None)
            # keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符,
            self.h_fcdrop1 = tf.nn.dropout(self.h_fcRelu1, self.keep_prob)

        # 输出层
        with tf.name_scope('fc2'):
            self.W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name = 'self.W_fc2')
            tf.summary.histogram('fc2 weights', self.W_fc2)
            # 偏移量
            self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]),name = 'self.b_fc2')
            tf.summary.histogram('fc2 biases', self.b_fc2)
            # 矩阵相乘 得到输出
            # self.y_conv = tf.matmul(self.h_fcdrop1, self.W_fc2) + self.b_fc2
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fcdrop1, self.W_fc2) + self.b_fc2, name='y_conv')

        # 计算代价
        with tf.name_scope('loss'):
            # tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
            # logits : 就是神经网络最后一层的输出
            # labels : 实际的标签
            # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_label, logits=self.y_conv)
            self.cross_entropy = - tf.reduce_sum(self.y_label* tf.log(self.y_conv),name = 'cross_entropy')
            # tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
            # 求取平均值
            # self.cross_entropy = tf.reduce_mean(self.cross_entropy,name='self.cross_entropy')
            tf.summary.scalar('loss', self.cross_entropy)

        # 梯度求解
        with tf.name_scope('adam_optimizer'):
            # Adam优化算法：是一个寻找全局最优点的优化算法。
            # 寻找误差最小的点
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            # 计算误差率
            self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_label, 1))
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)