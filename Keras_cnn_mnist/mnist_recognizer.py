"""
sundada
用keras去搭建一个神经网络
mnist识别器
mnist数据集作为机器学习中一个十分经典的数据集
Keras已经做好了它的获取方式
"""
from keras.datasets import mnist  # 运行代码的时候如果没有数据集自动下载到本地
import matplotlib.pyplot as plt
import numpy as np
# import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD

# from keras.optimizers import SGD  版本问题

# 获取数据
# keras封装的数据集模块中有一个loaddata函数，调用返回训练集X_train,Y_train，测试集X_test,Y_test
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 数据为numpy的ndarray类型
print("x_train_sharp" + str(X_train.shape))  # 60000*28*28的数据
print("y_train_sharp" + str(Y_train.shape))  # 六万元素的数组，代表类别
print("x_test_sharp" + str(X_test.shape))
print("y_test_sharp" + str(Y_test.shape))
# 绘制一张图片查看
plt.imshow(X_train[0], cmap='gray')  # 选取第一张图片，绘图模式为灰度图
plt.show()
# 如果要把一个图片送入神经网络，需要把像素扯出来变为数组，全连接层
# ndarray的reshape函数可以改变数组的形状。
X_train = X_train.reshape(60000, 784) / 255.0  # 归一化操作，除以灰度值的最大255，缩小范围，有利于梯度下降的进行
X_test = X_test.reshape(10000, 784) / 255.0

Y_train = to_categorical(Y_train, 10)  # 此函数可以方便的把一个数字变为one_hot编码
Y_test = to_categorical(Y_test, 10)
"""
豆豆问题是一个二分类问题，分类标签数据用一个数就好，0，1表示两类，输出层用一个神经元
因此mnist输出层为10个神经元，标签数据用one_hot编码的格式
"""
model = Sequential()  # 用来堆叠神经网络的载体，神经元堆叠在一起形成一个神经网络预测模型
# 神经元数量，激活函数类型，输入特征的维度
model.add(Dense(units=256, activation='relu', input_dim=784))  # 将dense堆叠到数列上
model.add(Dense(units=256, activation='relu'))  # 堆叠一个输出层的单神经元dense,汇合隐藏层的两个神经元，keras会自动求导
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
# 最终的输出预测值实际上是一个概率值,softmax，十个类输出有大有小，总概率为1，概率最大的那个是结果

# 以上神经网络创建完成
# 配置模型,指定使用什么样的代价函数，使用哪种梯度下降算法等
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
# 使用均方误差代价函数（对于分类问题不太适合，改为交叉熵代价函数）， loss:损失函数，代价函数
# optimizer:优化器，默认学习率0.01，优化算法，随机梯度下降算法，metrics：训练时侯得到的评估标准。指定为accuracy准确度
model.fit(X_train, Y_train, epochs=5000, batch_size=4096)  # fit函数开始训练
# model.fit(输入的样本数据, 标准答案数据,训练的回合数, 每次训练使用的样本数量)
# 训练完成之后用evaluate函数评估训练效果,函数返回代价损失和准确率
loss, accuracy = model.evaluate(X_test, Y_test)
# 输出代价损失，准确率
print("loss" + str(loss))
print("accuracy" + str(accuracy))
# 拓展，可以用一些正则化的手段减少过拟合
