"""
用keras去搭建一个神经网络
"""
import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense

# 获取数据
m = 100
X, Y = dataset.get_beans1(m)  # 第五课
plot_utils.show_scatter(X, Y)

model = Sequential()  # 用来堆叠神经网络的载体，神经元堆叠在一起形成一个神经网络预测模型
dense = Dense(units=1, activation='sigmoid', input_dim=1)  # 神经元数量，激活函数类型，输入特征的维度
model.add(dense)  # 将dense堆叠到数列上
# 以上神经网络创建完成
# 配置模型,指定使用什么样的代价函数，使用哪种梯度下降算法等
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# 使用均方误差代价函数 loss:损失函数，代价函数 optimizer:优化器，优化算法，随机梯度下降算法，metrics：训练时侯得到的评估标准。指定为accuracy准确度
model.fit(X, Y, epochs=5000, batch_size=10)  # fit函数开始训练
# model.fit(输入的样本数据, 标准答案数据,训练的回合数, 每次训练使用的样本数量)
# 全部样本上完成一次训练为一个回合
pres = model.predict(X)  # 做一次预测
plot_utils.show_scatter_curve(X, Y, pres)  # 同时绘制散点图和预测曲线

"""
============================================================
"""
# 用keras加入隐藏层去预测复杂的豆豆数据
