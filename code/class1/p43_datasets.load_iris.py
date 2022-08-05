#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Author:kela Time:2021/7/17

from sklearn import datasets
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data  # .data 返回 iris 数据集所有输入特征
y_data = datasets.load_iris().target  # .target 返回 iris 数据集所有标签
print("x_data from datasets: \n", x_data)
print("y_data from datasets: \n", y_data)

# 为表格增加行索引（左侧）和列标签（上方）
x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
print("x_data add index: \n", x_data)

x_data['类别'] = y_data  # 新加一列，列标签为‘类别’，数据为 y_data
print("x_data add a column: \n", x_data)

# 类型维度不确定时，建议用 print() 函数打印出来确认效果