from face_algorithm.siam_channel_CNN.model import Siam_Channel_Model, Siam_Model
from lfw import getNegPairsImg, getPosPairsImg
import numpy as np
import cv2
import pickle
import pandas as pd
from detect_align import findAlignFace_dlib
from sklearn.preprocessing import OneHotEncoder
import keras
from face_algorithm.webface import loadPairsWebface
from keras.utils.np_utils import to_categorical
import h5py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 从LFW pairs中生成siam-cnn训练数据集，仅作为跑通模型试验用
def createLFWSiamTrainData(imgSize, saveFileName):

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
        img1 = cv2.resize(img1, (imgSize, imgSize))
        img2 = cv2.resize(img2, (imgSize, imgSize))
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
        img1 = cv2.resize(img1, (imgSize, imgSize))
        img2 = cv2.resize(img2, (imgSize, imgSize))
        trainPair1.append(img1)
        trainPair2.append(img2)
        label.append(-1)

    f = h5py.File(saveFileName, 'w')
    f['train_pair1'] = np.array(trainPair1)
    f['train_pair2'] = np.array(trainPair2)
    f['label'] = np.array(label)
    f.close()

    #return np.array(trainPair1), np.array(trainPair2), np.array(label)

# 在LFW上测试siam-cnn
def siamLFWTest(imgSize, modelPath):

    siamCnn = Siam_Model(imgSize, load=True, loadModelPath=modelPath)

    acc = 0
    count = 0

    posScore = []
    negScore = []

    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        count += 1
        if count > 10:
            break
        try:
            img1 = findAlignFace_dlib(img1, imgSize)
            img2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        img1 = cv2.resize(img1, (imgSize, imgSize))/255.0
        img2 = cv2.resize(img2, (imgSize, imgSize))/255.0
        img1 = img1[np.newaxis, :]
        img2 = img2[np.newaxis, :]
        score = siamCnn.inference(img1, img2)
        print(score)
        # if score > 0:
        #     acc += 1
        # posScore.append(score)

    print("_____________________________________")
    count = 0
    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        count += 1
        if count > 10:
            break
        try:
            img1 = findAlignFace_dlib(img1, imgSize)
            img2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        img1 = cv2.resize(img1, (imgSize, imgSize))/255.0
        img2 = cv2.resize(img2, (imgSize, imgSize))/255.0
        img1 = img1[np.newaxis, :]
        img2 = img2[np.newaxis, :]
        score = siamCnn.inference(img1, img2)
        print(score)
        # if score <= 0:
        #     acc += 1
        # negScore.append(score)


    # posScoreFile = open('./data/siam_cnn_pos_score_file.pkl', 'wb')
    # pickle.dump(posScore, posScoreFile)
    # negScoreFile = open('./data/siam_cnn_neg_score_file.pkl', 'wb')
    # pickle.dump(negScore, negScoreFile)
    # print("lfw acc:", acc / (len(posScore) + len(negScore)))
    # print("siam channel model score save finished!")

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
    webfacePairsFileName = '/disk1/zhangxu_new/webface_pairs.h5'
    lfwPairsFileName = '/disk1/zhangxu_new/lfw_pairs.h5'

    #createLFWSiamTrainData(imgSize, lfwPairsFileName)
    train1, train2, label = loadPairsWebface(webfacePairsFileName)
    print(train1.shape)
    print(train2.shape)
    print(label.shape)
    train1 = train1 / 255.0
    train2 = train2 / 255.0

    # print(label)
    # for i in range(len(label)):
    #     if label[i] == -1:
    #         label[i] = 0
    #categorical_labels = to_categorical(label, num_classes=2)
    #print(categorical_labels)



    #siamCnn = Siam_Channel_Model(imgSize, load=False, loadModelPath=None)
    siamCnn = Siam_Model(imgSize, load=False, loadModelPath=None)
    #siamCnn = Siam_Channel_Model(imgSize, load=True, loadModelPath=modelPath)
    #siamCnn = Siam_Model(imgSize, load=True, loadModelPath=modelPath)

    siamCnn.train(train1, train2, label, epoch=5, batchSize=256)
    siamCnn.saveModel(modelPath)

    siamLFWTest(imgSize, modelPath)
    #plotJointBayesScore(posScorePath, negScorePath)
