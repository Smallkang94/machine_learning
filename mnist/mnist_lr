# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:54:05 2020

@author: Smallkang94
"""


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#数据加载
digit = load_digits()
data = digit.data
#数据探索，查看属性
print(digit.keys())
#查看第一幅图像，为8*8像素图像
print(digit.images[0])   #将图片转化为像素矩阵，方法二：print(digit.data[0].reshape((8,8)))
print(data[0])   #将第一幅图片像素矩阵展开成一行
print(data.shape)   #共有1797行
#第一张图片代表的数字
print(digit.target[0])
print(digit.target.shape) #共有1797行
#数据集所有标签值
print(digit.target_names)
#数据集描述
print(digit.DESCR)   #作者、数据来源等

#数据可视化
plt.gray() 
plt.imshow(digit.images[0])   #将第一张图片显示粗来
plt.show()

#分割数据
train_x, test_x, train_y, test_y = train_test_split(data, digit.target, test_size=0.25, random_state=33)

#使用Z-Score进行规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#创建LR分类器
lr = LogisticRegression()
lr.fit(train_ss_x, train_y)   #lr分类器使用训练集进行训练
predict_y = lr.predict(test_ss_x) #对测试集进行预测
print('LR模型准确率为{}'.format(accuracy_score(predict_y, test_y)))


