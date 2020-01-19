# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:29:13 2020

@author: Smallkang94
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#数据加载
data = pd.read_csv('team_cluster_data.csv', encoding='gbk')
train_x = data[["2019国际排名", "2018世界杯排名", "2015亚洲杯排名"]]
#print(train_x)

#数据规范化到[0, 1]区间
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
#print(train_x)

#使用手肘法确定簇的最佳数量
see = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    see.append(kmeans.inertia_)   #簇内误差平方和
x = range(1,11)
plt.xlabel('K')
plt.ylabel('SEE')
plt.plot(x, see, 'o-')
plt.show()

#创建K-Means聚类器
kmeans = KMeans(n_clusters=3)   #n_clusters为根据前面肘法确定，分成5簇
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)

#合并聚类结果，插入到原数据表中
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
#print(result)
result.rename({0: u'聚类结果'}, axis=1, inplace=True)   #新插入的列默认列名为0，需要更改列名
print(result)

#将结果导出到csv文件中
result.to_csv('team_cluster_result.csv')




