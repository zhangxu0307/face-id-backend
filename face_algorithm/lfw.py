import numpy as np
import pandas as pd
import cv2
# #from face_id import getRep_openface
#from vgg_face import getRep_VGGface
# from light_cnn_tf import getRep_lightCNN
# #from facenet_tf import getRep_facenet_tf
import matplotlib
matplotlib.use('Agg')
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
def plotSimliarityHist(rootPath):

    datafilePath = glob(rootPath+"*.csv")

    for filePath in datafilePath[:]: # 此处存在一个bug，hist图会产生交叠现象，目前是手工一个个的产生
        filename = filePath[7:-4]
        print(filename)
        data = pd.read_csv(filePath)
        print(data)
        hist = data["0"].hist()
        fig = hist.get_figure()
        fig.savefig('./data/' + filename + ".jpg")
        break

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

    #dirlist = sorted(os.listdir(datasetRoot))

    # for dir in dirlist:
    #     for root, d , files in os.walk(datasetRoot+dir):
    #         for file in files:
    #             imgPath = datasetRoot+dir+"/"+file
    #             print(imgPath)
    #             img = cv2.imread(imgPath)
    #             rep = getRep(img)

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

    # posFile = open('./data/lfw_pos_'+modelName+'.pkl', 'wb')
    # pickle.dump(posVec, posFile)
    # print("lfw pos rep extraction finished!")

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
    # negFile = open('./data/lfw_neg_'+modelName+'.pkl', 'wb')
    # pickle.dump(negVec, negFile)
    # print("lfw neg rep extraction finished!")



if __name__ == '__main__':

    #modelName = "VGGface"
    modelName = "openface"

    #modelName = "lightCNN"
    #modelName = "facenet"
    #runLFW(modelName)



    #rootPath = "./data/"
    # plotSimliarityHist(rootPath)

    # lfwRoot = "./data/lfw/"
    createLFWFeatureVec(modelName)

    # pkl_file = open('./data/lfw_pos_'+modelName+'.pkl', 'rb')
    # data1 = pickle.load(pkl_file)
    # print(len(data1))
















