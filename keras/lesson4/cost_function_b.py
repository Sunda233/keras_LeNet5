"""
sundada
2022年9月7日17点32分
预测函数y=w0变为y=w0+b
引入b绘制代价函数曲面，看看是不是一个扁扁的碗
代价函数从二维变成三维，给b留出一个维度
在w和b两个方向上分别求导，得到曲面某点的梯度进行梯度下降，拟合数据。
"""
import matplotlib.pyplot as plt
import dataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 进行3d绘图
# 从数据中获取随机豆豆
m = 100
xs, ys = dataset.get_beans(m)
# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')
# 豆豆毒性散点图
plt.scatter(xs, ys)

# 预测函数
w = 0.1
b = 0.1
y_pre = w * xs + b  # 没有b截距的话无法拟合
# 预测函数图像
plt.plot(xs, y_pre)
# 显示图像
plt.show()
# 代价函数
ws = np.arange(-1, 2, 0.1)  # 用arrange函数生成不同的w值与b值
bs = np.arange(-2, 2, 0.1)
# 将Axes3D和matplotlib做一个连接
fig = plt.figure()  # 得到plt的图形对象
ax = Axes3D(fig)  # 用fig对象创建一个Axes3D对象
ax.set_zlim(0, 2)  # 限定取值范围
for w in ws:  # 每次取不同的w
    es = []
    for b in bs:
        y_pre = w * xs + b
        # 得到w和b的关系
        e = (1 / m) * np.sum((ys - y_pre) ** 2)  # 计算均方误差
        es.append(e)  # 误差放入数组中
    # plt.plot(ws,es)
    figure = ax.plot(bs, es, w, zdir='y')  # 和plt的函数相比可以接受第三个维度的参数，zdir='y'把z轴的朝向换到y上面来
# 显示图像
plt.show()
