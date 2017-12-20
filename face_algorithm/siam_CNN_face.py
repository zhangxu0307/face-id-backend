from siam_channel_CNN.model import Siam_Channel_Model
from lfw import getNegPairsImg, getPosPairsImg
import numpy as np
import cv2
import pickle
import pandas as pd
from detect_align import findAlignFace_dlib
import keras

# 从LFW pairs中生成siam-cnn训练数据集，仅作为跑通模型试验用
def createLFWSiamTrainData(imgSize):

    trainPair1 = []
    trainPair2 = []
    label = []
    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        try:
            img1 = findAlignFace_dlib(img1, imgSize)
            img2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        img1 = cv2.resize(img1, (imgSize, imgSize))/255.0
        img2 = cv2.resize(img2, (imgSize, imgSize))/255.0
        trainPair1.append(img1)
        trainPair2.append(img2)
        label.append(1)

    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        try:
            img1 = findAlignFace_dlib(img1, imgSize)
            img2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        img1 = cv2.resize(img1, (imgSize, imgSize))/255.0
        img2 = cv2.resize(img2, (imgSize, imgSize))/255.0
        trainPair1.append(img1)
        trainPair2.append(img2)
        label.append(-1)

    return np.array(trainPair1), np.array(trainPair2), np.array(label)

# 在LFW上测试siam-cnn
def siamLFWTest(imgSize, modelPath):

    siamCnn = Siam_Channel_Model(imgSize, load=True, loadModelPath=modelPath)

    acc = 0

    posScore = []
    negScore = []

    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        try:
            img1 = findAlignFace_dlib(img1, imgSize)
            img2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        img1 = cv2.resize(img1, (imgSize, imgSize))/255.0
        img2 = cv2.resize(img2, (imgSize, imgSize))/255.0
        img1 = img1[np.newaxis, :]
        img2 = img2[np.newaxis, :]
        score = siamCnn.inference(img1, img2)[0][0]
        print(score)
        if score > 0:
            acc += 1
        posScore.append(score)

    print("_____________________________________")

    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        try:
            img1 = findAlignFace_dlib(img1, imgSize)
            img2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        img1 = cv2.resize(img1, (imgSize, imgSize))/255.0
        img2 = cv2.resize(img2, (imgSize, imgSize))/255.0
        img1 = img1[np.newaxis, :]
        img2 = img2[np.newaxis, :]
        score = siamCnn.inference(img1, img2)[0][0]
        print(score)
        if score <= 0:
            acc += 1
        negScore.append(score)


    posScoreFile = open('./data/siam_cnn_pos_score_file.pkl', 'wb')
    pickle.dump(posScore, posScoreFile)
    negScoreFile = open('./data/siam_cnn_neg_score_file.pkl', 'wb')
    pickle.dump(negScore, negScoreFile)
    print("lfw acc:", acc / (len(posScore) + len(negScore)))
    print("siam channel model score save finished!")

# 绘制siam-channel-model得分分布直方图
def plotJointBayesScore(posScorFilePath, negScorFilePath):

    posScoreFile = open(posScorFilePath, 'rb')
    posScore = pickle.load(posScoreFile)

    negScoreFile = open(negScorFilePath, 'rb')
    negScore = pickle.load(negScoreFile)

    pos = pd.Series(posScore)
    neg = pd.Series(negScore)

    hist1 = pos.hist()
    fig1 = hist1.get_figure()
    fig1.savefig('./data/siam_cnn_pos_score.jpg')

    hist2 = neg.hist()
    fig2 = hist2.get_figure()
    fig2.savefig('./data/siam_cnn_neg_score.jpg')

if __name__ == '__main__':

    imgSize = 128
    modelPath = "./models/siam_cnn.h5"
    lfwPosVecPath = './data/lfw_pos_openface.pkl'
    lfwNegVecPath = './data/lfw_neg_openface.pkl'
    posScorePath = './data/siam_cnn_pos_score_file.pkl'
    negScorePath = './data/siam_cnn_neg_score_file.pkl'

    train1, train2, label = createLFWSiamTrainData(imgSize)
    print(train1.shape)
    print(train2.shape)
    print(label.shape)

    siamCnn = Siam_Channel_Model(imgSize, load=False, loadModelPath=None)
    siamCnn.train(train1, train2, label, epoch=15, batchSize=64)
    siamCnn.saveModel(modelPath)

    siamLFWTest(imgSize, modelPath)
    plotJointBayesScore(posScorePath, negScorePath)
