"""
矩阵与向量计算
上节课代码。输入、权重系数、偏置系数都变成向量的形式（数组和多维数组）
一般用大写的字母表示向量和矩阵
"""

import numpy as np
import dataset
import plot_utils

# 获取豆豆数据
m = 100
X, Y = dataset.get_beans(m)  # 返回的是一个数组，不需要做额外的处理
print(X, Y)
# 用三维散点图画出来
plot_utils.show_scatter(X, Y)

# 编写预测模型
# w1 = 0.1
# w2 = 0.1
W = np.array([0.1, 0.1])  # 神经元与两个输入的权重系数
B = np.array([0.1])  # 神经元的偏置参数


# 编写前向传播代码

def forword_propgation(X):  # 同时包含豆豆的两个特征维度数据，此处X是参数名字，与上方X不同
    # z = w1 * x1s + w2 * x2s + b
    # 得到模型线性函数运算结果Z
    Z = X.dot(W.T) + B  # 改成向量运算 ，dot函数：点乘运算（行列对应元素相乘再相加），.T转置运算
    # 放入sigmoid中
    A = 1 / (1 + np.exp(-Z))  # exp操作也有广播机制,计算结果还是向量
    return A


# plot_utils.show_scatter_surface(X, Y, forword_propgation)  # 封装函数，同时画出豆豆的散点图和预测曲面

# 采用反向传播和随机梯度下降训练代码
for _ in range(500):
    for i in range(m):
        Xi = X[i]  # 现在x表示一行两列的数组
        Yi = Y[i]

        A = forword_propgation(Xi)  # 返回一行一列数组，表示豆豆样本最终的预测值
        # 计算误差代价，反向传播，通过梯度下降调整参数
        E = (Yi - A) ** 2
        # 都改为对向量求导，注意求导过程中矩阵的形状
        dEdA = -2 * (Yi - A)  # 一行一列
        dAdZ = A * (1 - A)  # 一行一列
        dZdW = Xi  # 一行两列
        dZdB = 1  # 一行一列

        dEdW = dEdA * dAdZ * dZdW
        dEdB = dEdA * dAdZ * dZdB

        alpha = 0.01
        W= W - alpha * dEdW
        B = B - alpha * dEdB

plot_utils.show_scatter_surface(X, Y, forword_propgation)
