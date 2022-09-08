"""
sundada
激活函数
将sigmoid激活函数带入到激活模型中
对激活模型进行梯度下降
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
w = 0.1  # 预测参数
b = 0.1
z = w * xs  # y_pre变为激活函数
# 编写预测函数 y=0.5*x
# sigmoid（z）=1/1+e^-z
a = 1 / (1 + np.exp(-z))  # 自然常数e为底的指数函数
plt.plot(xs, a)  # 绘制一条线
plt.show()  # 绘制图像
for _ in range(5000):  # 全部样本上做500次梯度下降，训练500次
    # 使用随机梯度下降算法完成参数的自动调整，梯度下降算法是在单个样本的代价函数上进行的。
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 对w和b求偏导
        # 带入激活函数
        z = w * x + b
        a = 1 / (1 + np.exp(-z))
        e = (y - a) ** 2
        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw = x
        dedw = deda * dadz * dzdw
        dzdb = 1
        dedb = deda * dadz * dzdb
        alpha = 0.05
        w = w - alpha * dedw
        b = b - alpha * dedb
    # 动态图像
    if _ % 100 == 0:  # 将绘图的频率降低100倍
        plt.clf()  # 清空图像
        pl.scatter(xs, ys)  # 绘制散点图
        z = w * xs + b
        a = 1 / (1 + np.exp(-z))
        plt.xlim(0, 1)  # 限制坐标的范围
        plt.ylim(0, 1.2)
        plt.plot(xs, a)  # 绘制一条线
        plt.pause(0.01)  # 暂停函数，暂停0.01秒
plt.show()  # 绘制图像
