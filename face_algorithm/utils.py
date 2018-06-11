import numpy as np
import pandas as pd
from face_algorithm.id_utils import saveFeatureVec,  loadFeatureVec


# 随机生成一组特征向量用于时间测试
def createTimeTestVec(sampleNum = 10000, featureDim = 2048):

    import time
    testDF = pd.DataFrame()
    testDFPath1 = "../media/test_vec.pkl"
    testDFPath2 = "../media/test_vec.h5"

    print("test random vec is generating.... ")
    for i in range(0, sampleNum):
        featureVec = np.random.random((featureDim, ))
        addVecSeries = pd.Series([featureVec], index=[i])
        testDF = pd.concat([testDF, addVecSeries], axis=0)
    print("test random vec finished!")

    saveFeatureVec(testDF, testDFPath1, format="pkl")

    saveFeatureVec(testDF, testDFPath2, format="h5")


if __name__ == '__main__':

    # 随机生成测试特征向量
    createTimeTestVec(sampleNum=100)

    # 测试加载特征向量时间
    import time

    t1 = time.time()

    candidateVec1 = loadFeatureVec("../media/test_vec.h5", format="h5")
    t2 = time.time()

    candidateVec2 = loadFeatureVec("../media/test_vec.pkl", format="pkl")
    t3 = time.time()

    print("h5 load time:", t2 - t1)
    print("pkl load time:", t3 - t2)