from face_algorithm.sphere_face_pt import getRep_SphereFace # 需要率先import，否则会core dumped
import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

import os
from matplotlib.pyplot import plot, savefig
from glob import glob
import pickle

# 计算成对的余弦相似度
def calcCosSimilarityPairs(rep1, rep2):
    return np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2))

# 获取负样本图像pair
def getNegPairsImg():

    negPairsTxt = pd.read_csv("./data/negative_pairs.txt", sep="   ", header=0)
    imgPath = pd.read_csv("./data/Path_lfw2.txt", header=-1)
    negPairsNum = len(negPairsTxt)
    print("neg pairs num:", negPairsNum)

    for i in range(negPairsNum):
        index1 = negPairsTxt.ix[i, "s1"]
        index2 = negPairsTxt.ix[i, "s2"]
        path1 = imgPath.ix[index1 - 1, 0]
        path2 = imgPath.ix[index2 - 1, 0]

        path1 = "./data/lfw/" + path1
        path2 = "./data/lfw/" + path2
        print(path1)
        print(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        yield img1,img2

# 获取正样本图像pair
def getPosPairsImg():

    posPairsTxt = pd.read_csv("./data/postive_pairs.txt", sep="   ", header=0)
    imgPath = pd.read_csv("./data/Path_lfw2.txt", header=-1)
    posPairsNum = len(posPairsTxt)
    print("pos pairs num:", posPairsNum)

    for i in range(posPairsNum):
        index1 = posPairsTxt.ix[i, "s1"]
        index2 = posPairsTxt.ix[i, "s2"]
        path1 = imgPath.ix[index1 - 1, 0]
        path2 = imgPath.ix[index2 - 1, 0]

        path1 = "./data/lfw/" + path1
        path2 = "./data/lfw/" + path2
        print(path1)
        print(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        yield img1, img2

# 测试LFW数据集相似度分布情况
def runLFW(modelName):

    acc = 0
    posScore = []
    negScore = []

    if modelName == "VGGface":
        from vgg_face import getRep_VGGface
        getRep = getRep_VGGface
    if modelName == "openface":
        from face_id import getRep_openface
        getRep = getRep_openface
    if modelName == "lightCNN":
        from light_cnn_tf import getRep_lightCNN
        getRep = getRep_lightCNN
    if modelName == "facenet":
        from facenet_tf import getRep_facenet_tf
        getRep = getRep_facenet_tf
    if modelName == "sphere_face":

        getRep = getRep_SphereFace

    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        try:
            rep1 = getRep(img1)
            rep2 = getRep(img2)
            score = calcCosSimilarityPairs(rep1, rep2)
        except:
            continue
        print(score)
        posScore.append(score)

    posCsv = pd.DataFrame(posScore)
    posCsv.to_csv("./data/pos_score_"+modelName+".csv", index=False)

    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        try:
            rep1 = getRep(img1)
            rep2 = getRep(img2)
            score = calcCosSimilarityPairs(rep1, rep2)
        except:
            continue
        print(score)
        negScore.append(score)

    negCsv = pd.DataFrame(negScore)
    negCsv.to_csv("./data/neg_score_"+modelName+".csv", index=False)

# 绘制相似度分布直方图
def plotSimliarityHist(modelName): # 此处仍有bug，两个直方图会有混叠现象，只能一个个绘制

    # 绘制负样本对得分
    filePath = "./data/neg_score_"+modelName+".csv"
    data = pd.read_csv(filePath)
    print(data)
    hist = data["0"].hist()
    fig1 = hist.get_figure()
    fig1.savefig('./data/neg_score_' + modelName + ".jpg")

    # 绘制正样本对得分
    filePath = "./data/pos_score_" + modelName + ".csv"
    data = pd.read_csv(filePath)
    print(data)
    hist = data["0"].hist()
    fig2 = hist.get_figure()
    fig2.savefig('./data/pos_score_' + modelName + ".jpg")

# 获取LFW数据中所有样本对的特征向量并保存成文件
def createLFWFeatureVec(modelName):

    if modelName == "VGGface":
        from vgg_face import getRep_VGGface
        getRep = getRep_VGGface
    if modelName == "openface":
        from face_id import getRep_openface
        getRep = getRep_openface
    if modelName == "lightCNN":
        from light_cnn_tf import getRep_lightCNN
        getRep = getRep_lightCNN
    if modelName == "facenet":
        from facenet_tf import getRep_facenet_tf
        getRep = getRep_facenet_tf

    posVec = []
    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        try:
            rep1 = getRep(img1)
            rep2 = getRep(img2)
            pairsVec = [rep1, rep2]
        except:
            continue
        posVec.append(pairsVec)

    posFile = open('./data/lfw_pos_'+modelName+'.pkl', 'wb')
    pickle.dump(posVec, posFile)
    print("lfw pos rep extraction finished!")

    negVec = []
    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        try:
            rep1 = getRep(img1)
            rep2 = getRep(img2)
            pairsVec = [rep1, rep2]
        except:
            continue
        negVec.append(pairsVec)
    negFile = open('./data/lfw_neg_'+modelName+'.pkl', 'wb')
    pickle.dump(negVec, negFile)
    print("lfw neg rep extraction finished!")


# 使用余弦相似度卡阈值计算准确率
def runLFWScore(modelName, threshold):

    acc = 0

    negFile = open('./data/lfw_neg_' + modelName + '.pkl', 'rb')
    negPairs = pickle.load(negFile)

    for pair in negPairs:
        x1 = pair[0]
        x2 = pair[1]
        score = calcCosSimilarityPairs(x1, x2)
        if score < threshold:
            acc += 1

    posFile = open('./data/lfw_pos_' + modelName + '.pkl', 'rb')
    posPairs = pickle.load(posFile)

    for pair in posPairs:
        x1 = pair[0]
        x2 = pair[1]
        score = calcCosSimilarityPairs(x1, x2)
        if score > threshold:
            acc += 1
    print("lfw cos classify acc:", acc/(len(posPairs)+len(negPairs)))


if __name__ == '__main__':

    #modelName = "VGGface"
    #modelName = "openface"
    #modelName = "lightCNN"
    #modelName = "facenet"
    modelName = "sphere_face"
    runLFW(modelName)

    #plotSimliarityHist(modelName)

    # lfwRoot = "./data/lfw/"
    #createLFWFeatureVec(modelName)

    # pkl_file = open('./data/lfw_pos_'+modelName+'.pkl', 'rb')
    # data1 = pickle.load(pkl_file)
    # print(len(data1))

    # threshold = 0.5
    # runLFWScore(modelName, threshold)
















