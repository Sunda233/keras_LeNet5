"""
sundada
2022年9月6日13点54分
 方差代价函数
 编写代价函数并绘制图像
 通过方差代价函数预测函数
 参数自适应调整方式，回归分析，代价函数
 一步到位的方式，但在输入特征过多，样本数量过大的时候却消耗计算资源
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
w = 0.1
y_pre = w * xs
# 编写预测函数 y=0.5*x
plt.plot(xs, y_pre)  # 绘制一条线
plt.show()  # 绘制图像
'''
# 计算误差
es = (ys - y_pre) ** 2
sum_e = np.sum(es)  # 求和函数
sum_e = (1 / 100) * sum_e
print(sum_e)
'''
ws = np.arange(0, 3, 0.1)  # 生成数组（起始值，结束值，递进大小）
# 搞一个循环试试用不同的w值
es = []
for w in ws:
    y_pre = w * xs
    e = (1 / 100) * np.sum((ys - y_pre) ** 2)
    print("W:" + str(w) + "E:" + str(e))  # 用str将浮点型转换为字符串值
    es.append(e)  # 每次w取不同误差的时候放入数组中

# 根据顶点坐标公式b/-2a,找到抛物线的最低点
w_min = np.sum(xs * ys) / np.sum(xs * xs)
print("e最小点的W:" + str(w_min))
# 最低点w值带回预测函数看看效果
y_pre = w_min * xs
# 配置图像
# 创建坐标系
plt.title("Size-Toxicity Function", fontsize=12)  # 设置图像的名称
plt.xlabel("Bean Size")  # 设置横坐标
plt.ylabel("Toxicity")  # 设置纵坐标
plt.plot(ws, es)
plt.show()  # 绘制图像

# 创建坐标系
plt.title("Size-Toxicity Function", fontsize=12)  # 设置图像的名称
plt.xlabel("w")  # 设置横坐标
plt.ylabel("e")  # 设置纵坐标
pl.scatter(xs, ys)  # 绘制散点图
plt.plot(xs, y_pre)
plt.show()  # 绘制图像
