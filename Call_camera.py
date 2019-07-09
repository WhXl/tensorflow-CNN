from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class ApplyCnn():
    def __init__(self):
        # 定义sess
        self.sess = tf.Session()
        #初始化变量
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        #导入预先处理好的模型
        #创造网络
        self.saver = tf.train.Saver
        self.saver = tf.train.import_meta_graph('my_net/model_save.ckpt.meta')
        #加载参数
        self.saver.restore(self.sess,'my_net/model_save.ckpt')
        self.graph = tf.get_default_graph()
        self.x_image = self.graph.get_tensor_by_name('x_image:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.y_conv = self.graph.get_tensor_by_name('y_conv:0')
    #识别单张图片
    def Imageprepare(self,path):
        self.path = path
        self.myimage = Image.open(self.path)
        self.myimage = self.myimage.resize((28, 28), Image.ANTIALIAS).convert('L')  # 变换成28*28像素，并转换成灰度图
        self.tv = list(self.myimage.getdata())  # 获取像素值
	print(tv)
        self.tva = [x/255.0 for x in self.tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑
        self.prediction = tf.argmax(self.y_conv, 1)
        self.predint = self.prediction.eval(feed_dict={self.x_image: [self.tva], self.keep_prob: 1.0}, session=self.sess)  # feed_dict输入数据给placeholder占位符
        print(self.predint[0])
    #识别多张图片
    def Num_imageprepare(self,path):
        self.path = path
        self.filelist = os.listdir(self.path)
        self.count = 0
        for self.files in self.filelist:
            self.Olddir = os.path.join(self.path, self.files);  # 原来的文件路径
            if os.path.isdir(self.Olddir):  # 如果是文件夹则跳过
                continue;
            self.f = os.path.basename(self.files)
            self.paths = os.path.join(self.path, self.f)
            self.myimage = Image.open(self.paths)
            self.myimage = self.myimage.resize((28, 28), Image.ANTIALIAS).convert('L')  # 变换成28*28像素，并转换成灰度图
            self.tv = list(self.myimage.getdata())  # 获取像素值
            print(len(tv))
            self.tva = [x / 255.0 for x in self.tv]
            self.prediction = tf.argmax(self.y_conv, 1)
            self.predint = self.prediction.eval(feed_dict={self.x_image: [self.tva], self.keep_prob: 1.0},
                                      session=self.sess)  # feed_dict输入数据给placeholder占位符
            print(self.predint[0])
            self.count = self.count + 1
        print('Total %d pictures' %(self.count))
    def Call_camera(self):
        self.cap = cv2.VideoCapture(0)  # 参数是0 表示打开内置摄像头
        while (1):
            self.ret, self.frame = self.cap.read()
            cv2.rectangle(self.frame, (270, 200), (370, 300), (0, 0, 255), 1)
            cv2.imshow("capture", self.frame)
            self.roiImg = self.frame[200:300, 270:370]
            self.img_resize = cv2.resize(self.roiImg, (28, 28), cv2.IMREAD_GRAYSCALE)
            self.img_gray = cv2.cvtColor(self.img_resize, cv2.COLOR_RGB2GRAY)
            # plt.imshow(img_gray)
            # plt.show()
            self.img = np.reshape(self.img_gray, [-1, 784])
            self.tv = self.img[0]
            # print(tv)
            self.tva = [x / 255.0 for x in self.tv]
            self.prediction = tf.arg_max(self.y_conv, 1)
            self.predint = self.prediction.eval(feed_dict={self.x_image: [self.tva], self.keep_prob: 1.0}, session=self.sess)
            print(self.predint[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ =='__main__':
    app = ApplyCnn()
    #识别单张图片
    path = 'Image/14.bmp'
    app.Imageprepare(path)
    #识别多张图片
    #path = 'Image'
    #app.Num_imageprepare(path)
    #调用摄像头
    #app.Call_camera()



# sess = tf.Session()
# init = tf.global_variables_initializer()
# saver = tf.train.Saver
# sess.run(init)
# saver = tf.train.import_meta_graph('my_net/model_save.ckpt.meta')
# saver.restore(sess, 'my_net/model_save.ckpt')
# graph = tf.get_default_graph()
# x = graph.get_tensor_by_name('x:0')
# keep_prob = graph.get_tensor_by_name('keep_prob:0')
# y_conv = graph.get_tensor_by_name('y_conv:0')

# 识别图片方法1
# def imageprepare(path):
#     myimage = Image.open(path)
#     myimage = myimage.resize((28, 28), Image.ANTIALIAS).convert('L')  #变换成28*28像素，并转换成灰度图
#     tv = list(myimage.getdata())  # 获取像素值
#     tva = [x/255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑
#     return tva
# path = 'Image/14.bmp'
# result = imageprepare(path)
# prediction = tf.argmax(y_conv, 1)
# predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符
# print(predint[0]) # 打印预测结果


##识别图片方法2
# file_name = 'Image/14.bmp'
# myimage = Image.open(file_name)
# myimage = myimage.resize((28, 28), Image.ANTIALIAS).convert('L')  # 变换成28*28像素，并转换成灰度图
# tv = list(myimage.getdata())  # 获取像素值
# tva = [x/255.0 for x in tv]
# prediction = tf.argmax(y_conv, 1)
# predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符
# print(predint[0])



# #识别多张图片
# def imageprepare(path):
#     filelist = os.listdir(path)
#     count = 0
#     for files in filelist:
#         Olddir = os.path.join(path, files);  # 原来的文件路径
#         if os.path.isdir(Olddir):  # 如果是文件夹则跳过
#             continue;
#         f = os.path.basename(files)
#         path = 'Image'
#         paths = os.path.join(path,f)
#         myimage = Image.open(paths)
#         myimage = myimage.resize((28, 28), Image.ANTIALIAS).convert('L')  # 变换成28*28像素，并转换成灰度图
#         tv = list(myimage.getdata())  # 获取像素值
#         # print(len(tv))
#         tva = [x / 255.0 for x in tv]
#         prediction = tf.argmax(y_conv, 1)
#         predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符
#         print(predint[0])
#         count = count +1
#     print(count)
# path = 'Image'
# result=imageprepare(path)


# 调用摄像头
# cap = cv2.VideoCapture(0)  # 参数是0 表示打开内置摄像头
# while(1):
#     ret, frame = cap.read()
#     # cv2.rectangle(frame, (270, 200), (370, 300), (0, 0, 255), 1)
#     cv2.imshow("capture", frame)
#     # roiImg = frame[200:300, 270:370]
#     img_resize = cv2.resize(frame, (28, 28), cv2.IMREAD_GRAYSCALE)
#     img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
#     # plt.imshow(img_gray)
#     # plt.show()
#     img = np.reshape(img_gray, [-1, 784])
#     tv = img[0]
#     # print(tv)
#     tva = [x/255.0 for x in tv]
#     prediction = tf.arg_max(y_conv, 1)
#     predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0}, session=sess)
#     print(predint[0])
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#          break
# cap.release()
# cv2.destroyAllWindows()
