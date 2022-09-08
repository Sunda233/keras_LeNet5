"""
卷积神经网络
采用mnist数据集的LeNet-5模型
"""
from keras.datasets import mnist  # 运行代码的时候如果没有数据集自动下载到本地
import matplotlib.pyplot as plt
import numpy as np
# import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv2D  # 引入二维卷积层
from keras.layers import AveragePooling2D  # 引入二维平均池化层
from keras.layers import Flatten  # 将池化层的输出平铺成一个数组

# from keras.optimizers import SGD  版本问题

# 获取数据
# keras封装的数据集模块中有一个loaddata函数，调用返回训练集X_train,Y_train，测试集X_test,Y_test
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 数据为numpy的ndarray类型

# 数据直接送入卷积层，卷积核在图像上做数据提取,明确指定图像有几个通道
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0  # 归一化操作，除以灰度值的最大255，缩小范围，有利于梯度下降的进行
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

Y_train = to_categorical(Y_train, 10)  # 此函数可以方便的把一个数字变为one_hot编码
Y_test = to_categorical(Y_test, 10)

model = Sequential()  # 用来堆叠神经网络的载体，神经元堆叠在一起形成一个神经网络预测模型
model.add(
    Conv2D(filters=6, kernel_size=(5, 5), strides=(2, 2), input_shape=(28, 28, 1), padding='valid', activation='relu'))
# model.add(Conv2D(卷积核数量=6,卷积核尺寸=(5,5),挪动步长=(1,1),输入形状=(28,28,1),填充模式='valid',激活函数='relu'))
# 卷积层中，输入数据是一个多维的张量，需要指定具体形状，才能构建卷积核，由输入形状Keras会构建合适的卷积核
# 【填充模式】越卷积越小，引入填充方案，现在原始图像周围填充几圈全是0的像素点，卷积完形状不变（same模式），不加填充的模式成为Valid模式
# 第一层卷积层结束，将结果（24*24*6）送入池化层，LeNet5使用的是窗口尺寸2*2的平均池化
model.add(AveragePooling2D(pool_size=(2, 2)))  # 指定大小 结果（12*12*6）
# 第二个卷积层
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
# 卷积完8*8*16，再添加池化层
model.add(AveragePooling2D(pool_size=(2, 2)))  # 指定大小 结果（4*4*16）
# 全连接层
model.add(Flatten())  # 池化层的输出平铺成为数组
model.add(Dense(units=120, activation='relu'))  # sigmoid,现采用relu
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))  # RBF 函数（即径向欧式距离函数）,目前已经被Softmax 取代
# LeNet5网络搭建完成
# 配置模型,指定使用什么样的代价函数，使用哪种梯度下降算法等
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
# 使用均方误差代价函数（对于分类问题不太适合，改为交叉熵代价函数）， loss:损失函数，代价函数
# optimizer:优化器，默认学习率0.01，优化算法，随机梯度下降算法，metrics：训练时侯得到的评估标准。指定为accuracy准确度
# model.fit(X_train, Y_train, epochs=5000, batch_size=4096)  # fit函数开始训练
model.fit(X_train, Y_train, epochs=100, batch_size=50)  # fit函数开始训练
# model.fit(输入的样本数据, 标准答案数据,训练的回合数, 每次训练使用的样本数量)
# 训练完成之后用evaluate函数评估训练效果,函数返回代价损失和准确率
loss, accuracy = model.evaluate(X_test, Y_test)
# 输出代价损失（交叉熵损失函数），准确率
print("当前代价损失+loss" + str(loss))
print("预测准确率+accuracy" + str(accuracy))
