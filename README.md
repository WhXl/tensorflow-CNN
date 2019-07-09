所需的环境

 tensorflow-gpu ==1.10 
	  numpy ==1.14.5 
  opencv-python == 4.0.0 
     Pillow-PIL ==0.1.dev0 
     matplotlib ==2.0.2

第一部分:简单介绍 

第一次上传的项目,最基础的神经网络,和其他的CNN不同之处主要有三个方面 :
   (1) 识别一张或多张图片
   (2) 调用摄像头实时识别数字
   (3) 使用tensorboard查看权重变化图,损失图,结构图等.

第二部分:快速开始 

1.复制这个项目
    git clone https://https://github.com/MrWXfaster/tensorflow-CNN.git
2.识别单张图片/多张图片/调用摄像头
解压,进入tensorflow-CNN 目录下,直接终端输入 python Call_camera.py就可以显示识别结果/其他的类似,因为写了中文注释,就不说了
3.模型可视化
在tensorflow-CNN 目录下,进入你所创建的环境,tensorboard --logdir log 点击显示的链接,就可以在浏览器中显示.

第三部分:如何训练
在tensorflow-CNN 目录下,进入你所创建的环境,python traincnn.py ,如果想要更改训练的步数,打开traincnn.py,更改即可.
