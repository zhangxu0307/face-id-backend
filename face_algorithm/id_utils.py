# from face_algorithm.sphere_face_pt import getRep_SphereFace
# 由于兼容性问题的存在，如果使用sphereface，打开此行注释，使用VGGface，保持此行注释
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from django.conf import settings
import cv2
import time



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
def calcCossimilarity(imgArr, candidate, getRep, verbose=False):

    # 特征向量维度
    if settings.METHOD == "VGGface":
        FEATURE_DIM = 2048
    if settings.METHOD == "sphereface":
        FEATURE_DIM = 512

    import time
    t1 = time.time()
    candidateArr = candidate.values # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist())) # 转化为numpy数组
    candidateArr = np.reshape(candidateArr, (-1, FEATURE_DIM))  # 防止出现一个人的情况

    t2 = time.time()
    testVec = getRep(imgArr)

    t3 = time.time()
    scoreMat = cosine_similarity(testVec, candidateArr)[0]  # 此处是个嵌套的array

    t4 = time.time()

    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[-1]
    t5 = time.time()

    if verbose:
        print("load and transfore time:", t2-t1)
        print("get feature rep time:", t3 - t2)
        print("calc cosine_similarity time:", t4 - t3)
        print("sort time:", t5 - t4)

    return resultID, scoreMat[sortIndex[-1]], testVec, candidateArr[sortIndex[-1], :] # 顺带返回特征向量，准备二次验证

# 计算欧氏距离
# def calcEuclidDistance(imgArr, candidate, getRep):
#     print(candidate)
#     candidateArr = candidate.values  # 传入参数是个dataframe
#     candidateArr = np.squeeze(np.array(candidateArr.tolist()))  # 转化为numpy数组
#     testVec = getRep(imgArr)
#     scoreMat = euclidean_distances(testVec, candidateArr)[0]  # 此处是个嵌套的array
#     #scoreMat = np.power(scoreMat, 2)
#     print(scoreMat)
#     sortIndex = np.argsort(scoreMat)
#     resultID = candidate.index[sortIndex].values[0]
#     return resultID, scoreMat[sortIndex[0]]

# 向特征文件中增加特征向量
def addFaceVec(imgArr, ID, getRep):

    addVec = getRep(imgArr)

    addVecSeries = pd.Series([addVec], index=[ID])
    settings.CANDIDATE = pd.concat([settings.CANDIDATE, addVecSeries], axis=0)
    print("current candidate vec:", settings.CANDIDATE)

    saveFeatureVec(settings.CANDIDATE, settings.CANDIDATEPATH, format="pkl")
    return

# 向特征文件中删除特征向量
def deleteFaceVec(ID):

    settings.CANDIDATE = settings.CANDIDATE.drop(ID)

    print("current candidate vec:", settings.CANDIDATE)

    saveFeatureVec(settings.CANDIDATE, settings.CANDIDATEPATH, format="pkl")
    return

# 将候选数组从pkl格式转化为h5格式
def transformPkl2HDF5():

    candidateVecPath = "../media/candidate_vec.pkl"
    candidateVecHDFPath = "../media/candidate_vec.h5"
    candidateVec = pd.read_pickle(candidateVecPath)
    candidateVec.to_hdf(candidateVecHDFPath, key="Gallery")


if __name__ == '__main__':

    # 测试计算相似度时间
    # from face_algorithm.vgg_face import getRep_VGGface
    # getRep = getRep_VGGface

    getRep = getRep_SphereFace

    candidateVec = loadFeatureVec("../media/candidate_vec.pkl", format="pkl")

    imgArr = cv2.imread("../media/MG1733013.jpg")
    t1 = time.time()
    calcCossimilarity(imgArr, candidateVec, getRep, verbose=True)
    t2 = time.time()
    print("calc cos time:", t2 - t1)

    # pkl文件转化为h5文件
    # transformPkl2HDF5()






