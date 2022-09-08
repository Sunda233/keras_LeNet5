"""
sundada
2022年9月6日14点32分
matplotlib绘制动态图像
随机梯度下降方式
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
y_pre = w * xs
# 编写预测函数 y=0.5*x
plt.plot(xs, y_pre)  # 绘制一条线
plt.show()  # 绘制图像
for _ in range(100):
    # 使用随机梯度下降算法完成参数的自动调整，梯度下降算法是在单个样本的代价函数上进行的。
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 代价函数系数 e=x^2*w^2+-2(xy)w+y^2
        # 系数a=x^2
        # b=-2xy
        # c=y^2
        # 斜率k=2aw+b
        k = 2 * (x ** 2) * w + (-2 * x * y)
        alpha = 0.1
        w = w - alpha * k  # 调整
        # 动态图像
        plt.clf()  # 清空图像
        pl.scatter(xs, ys)  # 绘制散点图
        plt.xlim(0, 1)  # 限制坐标的范围
        plt.ylim(0, 1.2)
        y_pre = w * xs
        plt.plot(xs, y_pre)  # 绘制一条线
        plt.pause(0.01)  # 暂停函数，暂停0.01秒
        # 在安装pyqt5后 matplotlib的pause（ ）函数无法正常退出  卸载后可以正常使用

# 配置图像
# 引入数据
# pl.scatter(xs, ys)  # 绘制散点图
# y_pre = w * xs
# # 编写预测函数 y=0.5*x
# plt.plot(xs, y_pre)  # 绘制一条线
# plt.show()  # 绘制图像
