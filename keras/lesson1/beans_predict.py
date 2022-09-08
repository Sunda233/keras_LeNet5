"""
2022年9月6日00点22分
numpy python的数学库
range 函数在python3里面生成一个可迭代的对象
Roseblatt 感知器模型
"""
import pylab as pl
import dataset  # 导入数据集模块,产生随机豆豆
from matplotlib import pyplot as plt  # 变灰色是没有调用该库的功能

# 豆豆直观绘图显示，引入matplotlib包
xs, ys = dataset.get_beans(100)  # 传入想要获得的数据的数量，获得豆豆大小xs和豆豆毒性ys
print(xs)
print(ys)  # 打印查看结果
# 配置图像
# 创建坐标系
plt.title("Size-Toxicity Function", fontsize=12)  # 设置图像的名称
plt.xlabel("Bean Size")  # 设置横坐标
plt.ylabel("Toxicity")  # 设置纵坐标
# 引入数据
pl.scatter(xs, ys)  # 绘制散点图
# 编写预测函数 y=0.5*x
w = 0.5
for m in range(100):  # 调整的过程来100次
    for i in range(100):  # 循环100个豆豆预测
        x = xs[i]
        y = ys[i]
        # 用预测函数预测一下豆豆毒性
        y_pre = w * x
        # 计算预测误差
        e = y - y_pre
        # 用误差调整w,防止大幅震荡的学习率alpha
        alpha = 0.05
        w = w + alpha * e * x
y_pre = w * xs  # 广播机制，自动将数组中的数与每个元素相乘
print(y_pre)
plt.plot(xs, y_pre)  # 绘制一条线
plt.show()  # 绘制图像
