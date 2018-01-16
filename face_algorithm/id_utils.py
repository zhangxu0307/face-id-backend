import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from django.conf import settings
import cv2


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

    import time
    t1 = time.time()
    candidateArr = candidate.values # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist())) # 转化为numpy数组
    candidateArr = np.reshape(candidateArr, (-1, 2048))  # 防止出现一个人的情况

    t2 = time.time()
    testVec = getRep(imgArr)

    t3 = time.time()
    scoreMat = cosine_similarity(testVec, candidateArr)[0] # 此处是个嵌套的array

    t4 = time.time()
    print(scoreMat)
    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[-1]
    t5 = time.time()

    print("load and transfore time:", t2-t1)
    print("get feature rep time:", t3 - t2)
    print("calc cosine_similarity time:", t4 - t3)
    print("sort time:", t5 - t4)

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

# 随机生成一组特征向量用于时间测试
def createTimeTestVec(sampleNum = 10000, featureDim = 2048):

    import time
    testDF = pd.DataFrame()
    testDFPath1 = "../media/test_vec.pkl"
    testDFPath2 = "../media/test_vec.h5"
    for i in range(0, sampleNum):
        print("%d is running..." %i)
        featureVec = np.random.random((featureDim, ))
        addVecSeries = pd.Series([featureVec], index=[i])
        testDF = pd.concat([testDF, addVecSeries], axis=0)
    print(testDF)
    t1 = time.time()
    saveFeatureVec(testDF, testDFPath1, format="pkl")
    t2 = time.time()
    saveFeatureVec(testDF, testDFPath2, format="h5")
    t3 = time.time()
    print(t2-t1)
    print(t3-t2)




if __name__ == '__main__':

    # pkl文件转化为h5文件
    #transformPkl2HDF5()

    # 随机生成测试特征向量
    # createTimeTestVec(sampleNum=20000)

    # 测试加载特征向量时间
    import time
    t1 = time.time()

    candidateVec1 = loadFeatureVec("../media/test_vec.h5", format="h5")
    t2 = time.time()

    candidateVec2 = loadFeatureVec("../media/test_vec.pkl", format="pkl")
    t3 = time.time()

    print(t2-t1)
    print(t3-t2)
    print(candidateVec1)
    print(candidateVec2)


    # 测试计算相似度时间
    from face_algorithm.vgg_face import getRep_VGGface
    getRep = getRep_VGGface
    imgArr = cv2.imread("../media/MG1633101.jpg")
    t4 = time.time()
    calcCossimilarity(imgArr, candidateVec1, getRep)
    t5 = time.time()
    print("calc cos time:", t5-t4)



