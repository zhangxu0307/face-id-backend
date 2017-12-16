import numpy as np
import pandas as pd
import cv2

# 计算成对的余弦相似度
def calcCosSimilarityPairs(rep1, rep2):
    return np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2))

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

if __name__ == '__main__':

    #g = getNegPairsImg()
    g = getPosPairsImg()
    for img1, img2 in g:
        print(img1.shape)













