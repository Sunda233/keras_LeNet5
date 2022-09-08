"""
sundada
创建一个具有隐藏层的神经网络
改造课5前向传播，增加神经元
权重参数W,偏置项B
"""
import pylab as pl
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans(100)
# 配置图像
# 创建坐标系
plt.title("Size-Toxicity Function", fontsize=12)  # 设置图像的名称
plt.xlabel("Bean Size")  # 设置横坐标
plt.ylabel("Toxicity")  # 设置纵坐标
# 引入数据
pl.scatter(xs, ys)  # 绘制散点图


# sigmoid封装函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 第一层
# 第一个神经元
# Numpy中的random.rand()主要用于返回一个或一组0到1之间的随机数或随机数组
w11_1 = np.random.rand()  # 第一个输入在第一个神经元在第一层,进行随机数初始化，有利于梯度下降的进行
b1_1 = np.random.rand()  # 第一层的第一个神经元的偏置项
# 第二个神经元
w12_1 = np.random.rand()
b2_1 = np.random.rand()
# 第二层第一个神经元，这一层的输入是上一层两个神经元的输出。
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()


# 前向传播
def forward_propgation(xs):
    # 第一层第一个神经元
    z1_1 = w11_1 * xs + b1_1  # 线性函数
    a1_1 = sigmoid(z1_1)  # 激活函数激活
    # 第一层第二个神经元
    z2_1 = w12_1 * xs + b2_1
    a2_1 = sigmoid(z2_1)
    # 第二层第一个神经元
    z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
    a1_2 = sigmoid(z1_2)  # 神经网络整体的预测输出
    return a1_2, z1_2, a2_1, z2_1, a1_1, z1_1


a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propgation(xs)
# 完成前向传播
plt.plot(xs, a1_2)  # 绘制一条线
plt.show()  # 绘制图像

for _ in range(5000):  # 全部样本上做500次梯度下降，训练500次
    # 使用随机梯度下降算法完成参数的自动调整，梯度下降算法是在单个样本的代价函数上进行的。
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 一次前向传播得到计算结果
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propgation(x)  # 送入单个样本x而不是全部样本xs
        # 实现反向传播
        # 写出代价函数e
        e = (y - a1_2) ** 2
        # 重点，如何把误差传播分配到每个权重参数和偏置项上。（使用链式法则）
        deda1_2 = -2*(y-a1_2)
        da1_2dz1_2 = a1_2*(1-a1_2)

        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1

        dedw11_2 = deda1_2*da1_2dz1_2*dz1_2dw11_2
        dedw21_2 = deda1_2*da1_2dz1_2*dz1_2dw21_2  # 将误差e传送到第二层神经元的两个权重参数

        dz1_2db1_2 = 1
        dedb1_2 = deda1_2*da1_2dz1_2*dz1_2db1_2

        # 再来看上一层，即隐藏层的两个神经元 ，继续进行反向传播
        dz1_2da1_1 = w11_2
        da1_1dz1_1 = a1_1*(1-a1_1)
        dz1_1dw11_1 = x
        dedw11_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1dw11_1  # e对第一层第一个神经元和第一个输入之间的权重w11_1的导数
        dz1_1db1_1 = 1
        dedb1_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1db1_1

        dz1_2da2_1 = w21_2
        da2_1dz2_1 = a2_1 * (1 - a2_1)
        dz2_1dw12_1 = x
        dedw12_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1 * da2_1dz2_1 * dz2_1dw12_1  # e对第一层第一个神经元和第一个输入之间的权重w11_1的导数
        dz2_1db2_1 = 1
        dedb2_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1 * da2_1dz2_1 * dz2_1db2_1
        # 到此为止把误差e传入到神经网络的所有参数上
        # 使用梯度下降算法调整参数
        alpha = 0.03
        w11_2 = w11_2 - alpha * dedw11_2
        w21_2 = w21_2 - alpha * dedw21_2
        b1_2 = b1_2 - alpha * dedb1_2

        w12_1 = w12_1 - alpha * dedw12_1
        b2_1 = b2_1 - alpha * dedb2_1

        w11_1 = w11_1 - alpha * dedw11_1
        b1_1 = b1_1 - alpha * dedb1_1

    # 动态图像
    if _ % 100 == 0:  # 将绘图的频率降低100倍
        plt.clf()  # 清空图像
        pl.scatter(xs, ys)  # 绘制散点图
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propgation(xs)
        # plt.xlim(0, 1)  # 限制坐标的范围
        # plt.ylim(0, 1.2)
        plt.plot(xs, a1_2)  # 绘制一条线
        plt.pause(0.01)  # 暂停函数，暂停0.01秒
plt.show()  # 绘制图像