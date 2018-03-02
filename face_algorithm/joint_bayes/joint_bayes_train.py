from face_algorithm.joint_bayes.joint_bayesian import *
from face_algorithm.webface import loadWebfaceVec
import pickle
import pandas as pd
import os

# 使用特征向量数据集进行joint bayes训练
def jointBayesTrain(trainFilePath, modelPath, vecName):

    trainx, trainy = loadWebfaceVec(trainFilePath)
    modelPath = os.path.join(modelPath, vecName)
    print("saved joint bayes model path:", modelPath)
    JointBayesian_Train(trainx, trainy, modelPath)

# lfw测试
def lfw_test(lfwPosVecPath, lfwNegVecPath, modelPath, vecName, posScorePath, negPosScorePath, threshold=-10):

    acc = 0
    modelPath = os.path.join(modelPath, vecName)
    print(modelPath)
    A_path = os.path.join(modelPath, "A.pkl")
    G_path = os.path.join(modelPath, "G.pkl")
    with open(A_path, "rb") as f:
        A = pickle.load(f)
    with open(G_path, "rb") as f:
        G = pickle.load(f)

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

    posScoreFile = open(posScorePath, 'wb')
    pickle.dump(posScore, posScoreFile)
    negScoreFile = open(negPosScorePath, 'wb')
    pickle.dump(negScore, negScoreFile)
    print("lfw acc:", acc/(len(lfwPosPairs)+len(lfwNegPairs)))
    print("joint bayes score save finished!")

# 绘制joint bayes得分分布直方图
def plotJointBayesScore(posScorFilePath, negScorFilePath):

    posScoreFile = open(posScorFilePath, 'rb')
    posScore = pickle.load(posScoreFile)

    negScoreFile = open(negScorFilePath, 'rb')
    negScore = pickle.load(negScoreFile)

    pos = pd.Series(posScore)
    neg = pd.Series(negScore)

    hist1 = pos.hist()
    fig1 = hist1.get_figure()
    fig1.savefig('../data/joint_bayes_pos_score.jpg')

    hist2 = neg.hist()
    fig2 = hist2.get_figure()
    fig2.savefig('../data/joint_bayes_neg_score.jpg')


if __name__ == "__main__":

    # 训练用特征向量文件，目前使用webface生成
    #trainFilePath = "/disk1/zhangxu_new/webface_vec_openface_v2.h5"
    #trainFilePath = "/disk1/zhangxu_new/webface_vec_VGGface.h5"
    trainFilePath = "/disk1/zhangxu_new/webface_vec_sphereface.h5"

    # 模型路径以及所用特征向量类型
    modelPath = "../models/joint_bayes/"
    vecName = "sphereface"

    # lfw对应特征向量文件
    lfwPosVecPath = '../data/lfw_pos_'+vecName+'.pkl'
    lfwNegVecPath = '../data/lfw_neg_'+vecName+'.pkl'

    # lfw 结果score文件
    lfwPosScorePath = '../data/joint_bayes_lfw_pos_score'+vecName+'.pkl'
    lfwNegScorePath = '../data/joint_bayes_lfw_neg_score' + vecName + '.pkl'

    jointBayesTrain(trainFilePath, modelPath, vecName)
    #lfw_test(lfwPosVecPath, lfwNegVecPath, modelPath, vecName, lfwPosScorePath, lfwNegScorePath)
    #plotJointBayesScore(lfwPosScorePath, lfwNegScorePath)