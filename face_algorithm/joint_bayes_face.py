#coding=utf-8
import sys
import numpy as np
#from joint_bayes.common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from face_algorithm.joint_bayes.joint_bayesian import *
from face_algorithm.webface import loadWebfaceVec
import pickle
import pandas as pd
from django.conf import settings

#modelPath = settings.BASE_DIR+"/face_algorithm/models/" # django工程路径写法
modelPath = "./models/"
with open(modelPath + "A.pkl", "rb") as f:
    A = pickle.load(f)
with open(modelPath + "G.pkl", "rb") as f:
    G = pickle.load(f)

# joint bayes训练
def jointBayesTrain(trainFilePath, modelPath):

    trainx, trainy = loadWebfaceVec(trainFilePath)

    JointBayesian_Train(trainx, trainy, modelPath)

# lfw测试
def lfw_test(lfwPosVecPath, lfwNegVecPath, modelPath, threshold=-10):

    acc = 0

    # with open(modelPath+"A.pkl", "rb") as f:
    #     A = pickle.load(f)
    # with open(modelPath+"G.pkl", "rb") as f:
    #     G = pickle.load(f)

    lfwPos_file = open(lfwPosVecPath, 'rb')
    lfwPosPairs = pickle.load(lfwPos_file)
    print("lfw pos pairs num:", len(lfwPosPairs))

    posScore = []
    negScore = []

    for pair in lfwPosPairs:
        x1 = pair[0]
        x2 = pair[1]
        score = Verify(A, G, x1, x2)
        print(score)
        if score > threshold:
            acc += 1
        posScore.append(score)

    print("_____________________________________")

    lfwNeg_file = open(lfwNegVecPath, 'rb')
    lfwNegPairs = pickle.load(lfwNeg_file)
    print("lfw neg pairs num:", len(lfwNegPairs))

    for pair in lfwNegPairs:
        x1 = pair[0]
        x2 = pair[1]
        score = Verify(A, G, x1, x2)
        print(score)
        if score <= threshold:
            acc += 1
        negScore.append(score)

    posScoreFile = open('./data/pos_score_file.pkl', 'wb')
    pickle.dump(posScore, posScoreFile)
    negScoreFile = open('./data/neg_score_file.pkl', 'wb')
    pickle.dump(negScore, negScoreFile)
    print("lfw acc:", acc/(len(lfwPosPairs)+len(lfwNegPairs)))
    print("joint bayes score save finished!")

# 绘制joint bayes得分分布直方图
def plotJointBayesScore(posScorFilePath, negScorFilePath):

    posScoreFile = open('./data/pos_score_file.pkl', 'rb')
    posScore = pickle.load(posScoreFile)

    negScoreFile = open('./data/neg_score_file.pkl', 'rb')
    negScore = pickle.load(negScoreFile)

    pos = pd.Series(posScore)
    neg = pd.Series(negScore)

    # hist1 = pos.hist()
    # fig1 = hist1.get_figure()
    # fig1.savefig('./data/joint_bayes_pos_score.jpg')

    hist2 = neg.hist()
    fig2 = hist2.get_figure()
    fig2.savefig('./data/joint_bayes_neg_score.jpg')


if __name__ == "__main__":

    trainFilePath = "/disk1/zhangxu_new/webface_vec_openface.h5"
    #modelPath = "./models/"
    lfwPosVecPath = './data/lfw_pos_openface.pkl'
    lfwNegVecPath = './data/lfw_neg_openface.pkl'

    # jointBayesTrain(trainFilePath, modelPath)
    lfw_test(lfwPosVecPath, lfwNegVecPath, modelPath)
    #plotJointBayesScore(lfwPosVecPath, lfwNegVecPath)