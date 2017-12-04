
import cv2
import os
import numpy as np
import pandas as pd
#np.set_printoptions(precision=2)

import openface
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

# 参数及模型加载
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


dlibModelPath = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
openfaceModelPath = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96

align = openface.AlignDlib(dlibModelPath)
net = openface.TorchNeuralNet(openfaceModelPath, imgDim)


# 获取人脸表示向量
def getRep(rgbImg):

    if rgbImg is None:
        raise Exception("Unable to load image")
    #rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face")

    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image")

    rep = net.forward(alignedFace)

    return rep


def calcCossimilarity(imgArr, candidate):

    candidateArr = candidate.values # 传入参数是个dataframe
    candidateArr = np.squeeze(np.array(candidateArr.tolist())) # 转化为numpy数组
    testVec = getRep(imgArr)
    scoreMat = cosine_similarity(testVec, candidateArr)[0] # 此处是个嵌套的array
    print(scoreMat.shape)
    sortIndex = np.argsort(scoreMat)
    resultID = candidate.index[sortIndex].values[0]
    print(resultID)
    return resultID, scoreMat[sortIndex[-1]]

def addFaceVec(imgArr, ID):

    addVec = getRep(imgArr)
    addVecSeries = pd.Series([addVec], index=[ID])
    settings.CANDIDATE = pd.concat([settings.CANDIDATE, addVecSeries], axis=0)
    print(settings.CANDIDATE)
    settings.CANDIDATE.to_pickle(settings.CANDIDATEPATH)
    return

def deleteFaceVec(ID):
    pass

if __name__ == '__main__':

    faceMat = []
    #settings.CANDIDATE

    imgPath1 = "../media/Aaron_Eckhart_0001.jpg"
    imgPath2 = "../media/Aaron_Guiel_0001.jpg"
    imgPath3 = "../media/Aaron_Peirsol_0001.jpg"
    imgPath4 = "../media/Aaron_Peirsol_0002.jpg"
    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)
    img3 = cv2.imread(imgPath3)
    img4 = cv2.imread(imgPath4)
    rep1 = getRep(img1)
    #print(rep1)
    rep2 = getRep(img2)
    #print(rep2)
    rep3 = getRep(img3)
    #print(rep3)

    addFaceVec(img1)
    addFaceVec(img2)
    addFaceVec(img3)

    index, similarity = calcCossimilarity(img4, faceMat)
    print(index, similarity)

