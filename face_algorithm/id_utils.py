import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from django.conf import settings
from .face_id import getRep_openface
from face_algorithm.vgg_face import getRep_VGGface


# 计算余弦相似度
def calcCossimilarity(imgArr, candidate):

    #print(candidate)

    candidateArr = candidate.values # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist())) # 转化为numpy数组
    testVec = getRep_openface(imgArr)
    #testVec = getRep_VGGface(imgArr)
    scoreMat = cosine_similarity(testVec, candidateArr)[0] # 此处是个嵌套的array
    print(scoreMat)
    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[-1]
    return resultID, scoreMat[sortIndex[-1]], testVec, candidateArr[sortIndex[-1], :] # 顺带返回特征向量，准备二次验证

# 计算欧氏距离
def calcEuclidDistance(imgArr, candidate):
    print(candidate)
    candidateArr = candidate.values  # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist()))  # 转化为numpy数组
    testVec = getRep_openface(imgArr)
    scoreMat = euclidean_distances(testVec, candidateArr)[0]  # 此处是个嵌套的array
    #scoreMat = np.power(scoreMat, 2)
    print(scoreMat)
    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[0]
    return resultID, scoreMat[sortIndex[0]]

# 向特征文件中增加特征向量
def addFaceVec(imgArr, ID):

    addVec = getRep_openface(imgArr)
    addVecSeries = pd.Series([addVec], index=[ID])
    settings.CANDIDATE = pd.concat([settings.CANDIDATE, addVecSeries], axis=0)
    print(settings.CANDIDATE)
    settings.CANDIDATE.to_pickle(settings.CANDIDATEPATH)
    return

# 向特征文件中删除特征向量
def deleteFaceVec(ID):

    settings.CANDIDATE = settings.CANDIDATE.drop(ID)
    print(settings.CANDIDATE)
    settings.CANDIDATE.to_pickle(settings.CANDIDATEPATH)
    return