import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import model


class TrainCnn():
    def __init__(self):
        # 创建模型
        self.network = model()

        self.sess = tf.InteractiveSession()
        # 可视化图
        self.writer = tf.summary.FileWriter("log/", self.sess.graph)
        # 可视化权重图
        self.merged = tf.summary.merge_all()
        # 获取MNIST数据集
        self.data = input_data.read_data_sets('Mnist_data', one_hot=True)

    def train(self):
        # 每次读取图片数量
        batch_size = 100
        # 定义sess
        self.sess = tf.Session()
        # 开始计算
        self.sess.run(tf.global_variables_initializer())
        #保存权重
        self.saver =tf.train.Saver()

        # 计算20000次
        for i in range(2000):
            batch = self.data.train.next_batch(batch_size)
            # 输出错误率
            if i % 100 == 0:
                realrate, cross = self.sess.run([self.network.accuracy, self.network.cross_entropy],
                                                feed_dict={self.network.x_image: batch[0],
                                                          self.network.y_label: batch[1], self.network.keep_prob: 0.5})
                # 显示出图表的变化
                result = self.sess.run(self.merged,feed_dict={self.network.x_image: batch[0],
                                                           self.network.y_label: batch[1], self.network.keep_prob: 0.5})
                self.writer.add_summary(result, i)

                print('step %d, training accuracy:%g ,cross_entropy:%g' % (i, realrate, cross))

            # 训练网络
            self.sess.run(self.network.train_step,
                          feed_dict={self.network.x_image: batch[0], self.network.y_label: batch[1],
                                     self.network.keep_prob: 0.5})
        #保存网络
        #self.saver.save(self.sess, "my_net/model_save.ckpt")

if __name__ == '__main__':
    app = TrainCnn()
    app.train()
