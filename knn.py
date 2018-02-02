# coding:utf8
import pandas as pd
import numpy as np
import datetime


def getTrainLabelAndMat():
    # 读取train.csv文件
    traincsv = pd.read_csv(r'digit recognizer\train.csv')
    # 取出label
    trainLabel0 = traincsv.values[0:, :1]
    # 取出训练数据
    trainMat = traincsv.values[0:, 1:]
    n = trainLabel0.shape[0]
    trainLabel = np.zeros(n)
    for i in range(0, n):
        trainLabel[i] = trainLabel0[i][0]
    # 返回训练标签和训练数据矩阵
    return trainLabel, trainMat


def getTestMat():
    testcsv = pd.read_csv(r'digit recognizer\test.csv')
    testMat = testcsv.values[0:, 0:]
    # 返回测试数据矩阵
    return testMat


# 分类函数，输入当前向量、训练矩阵，矩阵中向量对应标签，k值大小，返回类别
def classifyDigit(nowVect, dataSet, labels, k):
    # 得到训练矩阵行数
    dataSetSize = dataSet.shape[0]
    # 做差，平方，按行相加，开根
    diffMat = np.tile(nowVect, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)  # 按行相加，axis=0按列相加
    distances = sqDistance ** 0.5
    # 根据距离排序
    sortedDistIndicies = distances.argsort()
    # 统计前k近的点中各个类别的数量
    classCount = {}
    for i in range(k):
        nowlabel = labels[sortedDistIndicies[i]]
        classCount[nowlabel] = classCount.get(nowlabel, 0) + 1
    # 迭代一遍找到出现次数最多的类别
    maxCount = 0
    answer = ""
    for k, v in classCount.iteritems():
        if v > maxCount:
            maxCount = v
            answer = k
    return answer


def check(k):
    trainLabel, trainMat = getTrainLabelAndMat()
    testMat = getTestMat()
    # 储存结果的列表
    testLabel = []
    testId = range(1, testMat.shape[0] + 1)
    cnt = 0
    # 对每一个测试数据向量进行分类，并记录结果
    for nowVect in testMat:
        nowLabel = classifyDigit(nowVect, trainMat, trainLabel, k)
        testLabel.append(int(nowLabel))
        cnt += 1
        print cnt, int(nowLabel)
    # 按照给定格式生成csv文件
    dataframe = pd.DataFrame({'ImageId': testId, 'Label': testLabel})
    dataframe.to_csv("submissions.csv", index=False)


if __name__ == '__main__':
    start = datetime.datetime.now()
    check(3)
    end = datetime.datetime.now()
    print '使用时间：' + (end - start)
