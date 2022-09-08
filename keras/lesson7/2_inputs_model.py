"""
sundada
2022年9月7日23点56分
mat plotlib文挡地址:https://matplotlib.org/stable/tutorials/index.html
在课6的基础上增加一个输入，初始维度，豆豆大小，增加维度，颜色深浅
从一元一次函数改为二元一次函数，反向传播的时候多考虑一个权重参数。
3D绘图比较麻烦，封装了一个绘图库，直接调库就可以进行绘图
"""
import numpy as np
import dataset
import plot_utils

# 获取豆豆数据
m = 100
xs, ys = dataset.get_beans(m)
print(xs, ys)
# 用三维散点图画出来
plot_utils.show_scatter(xs, ys)

# 编写预测模型
w1 = 0.1
w2 = 0.1
b = 0.1
# 编写前向传播代码
# numpy数组的切片操作，先手动切成两部分，等下节课学完向量和数组再处理
x1s = xs[:, 0]  # 所有的行上把第0列切割形成一个新的数组
x2s = xs[:, 1]  # 第一列 ，得到豆豆大小和豆豆颜色深浅两组数据


def forword_propgation(x1s, x2s):
    z = w1 * x1s + w2 * x2s + b
    a = 1 / (1 + np.exp(-z))
    return a


plot_utils.show_scatter_surface(xs, ys, forword_propgation)  # 封装函数，同时画出豆豆的散点图和预测曲面

# 采用反向传播和随机梯度下降训练代码
for _ in range(500):
    for i in range(m):
        x = xs[i]  # 现在x表示一个二维的随机数组
        y = ys[i]
        x1 = x[0]
        x2 = x[1]

        a = forword_propgation(x1,x2)
        # 计算误差代价，反向传播，通过梯度下降调整参数
        e = (y-a)**2

        deda = -2*(y-a)
        dadz = a*(1-a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1

        dedw1= deda*dadz*dzdw1
        dedw2 = deda * dadz * dzdw2
        dedb = deda*dadz*dzdb

        alpha = 0.01
        w1 = w1-alpha*dedw1
        w2 = w2-alpha*dedw2
        b = b - alpha * dedb

plot_utils.show_scatter_surface(xs, ys, forword_propgation)

#
"""
拓展：加入隐藏层以后可以扭曲曲面和等高线的神经元网络，
像课6一样麻烦，所以使用矩阵和向量的概念，并且使用keras框架让编码变得简单
"""


