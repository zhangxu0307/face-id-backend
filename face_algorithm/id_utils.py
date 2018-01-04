import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from django.conf import settings


def saveFeatureVec(dataframe, filePath, format="pkl"):

    if format == "pkl":
        dataframe.to_pickle(filePath)
    if format == "h5":
        dataframe.to_hdf(filePath, key="Gallery")

def loadFeatureVec(filePath, format="pkl"):

    dataframe = None

    if format == "pkl":
        dataframe = pd.read_pickle(filePath)
    if format == "h5":
        dataframe = pd.read_hdf(filePath, key="Gallery")

    return dataframe


# 计算余弦相似度
def calcCossimilarity(imgArr, candidate, getRep):

    candidateArr = candidate.values # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist())) # 转化为numpy数组
    candidateArr = np.reshape(candidateArr, (-1, 2048))  # 防止出现一个人的情况
    testVec = getRep(imgArr)
    scoreMat = cosine_similarity(testVec, candidateArr)[0] # 此处是个嵌套的array
    print(scoreMat)
    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[-1]

    return resultID, scoreMat[sortIndex[-1]], testVec, candidateArr[sortIndex[-1], :] # 顺带返回特征向量，准备二次验证

# 计算欧氏距离
def calcEuclidDistance(imgArr, candidate, getRep):
    print(candidate)
    candidateArr = candidate.values  # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist()))  # 转化为numpy数组
    testVec = getRep(imgArr)
    scoreMat = euclidean_distances(testVec, candidateArr)[0]  # 此处是个嵌套的array
    #scoreMat = np.power(scoreMat, 2)
    print(scoreMat)
    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[0]
    return resultID, scoreMat[sortIndex[0]]

# 向特征文件中增加特征向量
def addFaceVec(imgArr, ID, getRep):

    addVec = getRep(imgArr)
    print(addVec.shape)
    addVecSeries = pd.Series([addVec], index=[ID])
    settings.CANDIDATE = pd.concat([settings.CANDIDATE, addVecSeries], axis=0)
    print(settings.CANDIDATE)
    #settings.CANDIDATE.to_pickle(settings.CANDIDATEPATH)
    saveFeatureVec(settings.CANDIDATE, settings.CANDIDATEPATH, format="pkl")
    return

# 向特征文件中删除特征向量
def deleteFaceVec(ID):

    settings.CANDIDATE = settings.CANDIDATE.drop(ID)
    print(settings.CANDIDATE)
    #settings.CANDIDATE.to_pickle(settings.CANDIDATEPATH)
    saveFeatureVec(settings.CANDIDATE, settings.CANDIDATEPATH, format="pkl")
    return

# 将候选数组从pkl格式转化为h5格式
def transformPkl2HDF5():

    candidateVecPath = "../media/candidate_vec.pkl"
    candidateVecHDFPath = "../media/candidate_vec.h5"
    candidateVec = pd.read_pickle(candidateVecPath)
    candidateVec.to_hdf(candidateVecHDFPath, key="Gallery")


if __name__ == '__main__':

    import time
    transformPkl2HDF5()
    t1 = time.time()
    #candidateVec = pd.read_hdf("../media/candidate_vec.h5")
    candidateVec = loadFeatureVec("../media/candidate_vec.h5", format="h5")
    t2 = time.time()
    #candidateVec = pd.read_pickle("../media/candidate_vec.pkl")
    candidateVec = loadFeatureVec("../media/candidate_vec.pkl", format="pkl")
    t3 = time.time()
    print(t2-t1)
    print(t3-t2)
    print(candidateVec)

