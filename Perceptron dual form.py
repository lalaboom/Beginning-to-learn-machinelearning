# -*- coding:utf-8 -*-
# Filename: train2.2.py
# Author：hankcs
# Date: 2015/1/31 15:15
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
training_set = np.array([[[3,3],1], [[4,3] , 1], [[1,1],-1], [[5,2],-1]])#前面两个元素是特征值，后面那个元素是分类
length= len(training_set)
a= np.zeros(len(training_set),np.float) #矩阵a的长度为训练集样本数，类型为float
b= 0.0 #偏置参数初始值为0
Gram = None #定义Gram矩阵
y= np.array(training_set[:,1]) #令y=分类集[1 1 -1 -1]
x= np.empty((len(training_set),2),np.float)#x为4*2的特征矩阵,注意初始时是空的
for i in range (len(training_set)):
    x[i] = training_set[i][0]#将训练集中特征值矩阵付给x矩阵
history = []#记录每次迭代结果

def cal_gram():  #计算Gram矩阵
    g= np.empty((len(training_set),len(training_set)), np.int)
    for i in range(len(training_set)):
        for j in range(len(training_set)):
            g[i][j]= np.dot(training_set[i][0],training_set[j][0]) #G=xi*xj
        return g
def updata(i):       #随机梯度下降更新参数


       global a,b
       a[i] += 1  #根据误分类点更新参数
       b= b + 1 *y[i] # 1 是学习速率
       history.append([np.dot( a * y , x ),b]) #history记录每次迭代结果
       print(a, b) #输出每次迭代结果

#计算yi(Gram*xi+b),用来判断是否是误分类点
def cal (i):
    global a,b,x,y
    res = np.dot(a * y, Gram[i])
    res = (res + b) * y[i] #返回
    return res

#检查是否已经正确分类
def check():
        global a,b,x,y
        flag = False
        for i in range(length):
            if cal(i) <= 0:
                flag = True
                updata(i)
        if not flag:
               w = np.dot((a * y ),x)   #计算w
               print("result: w:" + str(w) + "b:" + str(b))#输出最后结果
               return False
        return True

#if _name_ == "_main_":
if __name__ == '__main__':
       Gram = cal_gram() #初始化Gram矩阵
       for i in range(1000): #迭代次数
           if not check():break #如果已正确分类则结束循环
print("start")

