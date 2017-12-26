import numpy as np
import pandas as pd
import h5py
import os
import cv2
import copy
from face_algorithm.detect_align import findAlignFace_dlib

webfaceRoot = "/disk1/zhangxu_new/CASIA-WebFace/"


# 由webface数据集生成某种模型的特征向量集
def createWebfaceVec(modelName, saveFilePath):

    peopleNum = 5000 # 选取人数
    singleSubNum = 50 # 每人选取图片数
    datax = []
    datay = []

    if modelName == "VGGface":
        from vgg_face import getRep_VGGface
        getRep = getRep_VGGface
    if modelName == "openface":
        from face_id import getRep_openface
        getRep = getRep_openface
    if modelName == "lightCNN":
        from light_cnn_tf import getRep_lightCNN
        getRep = getRep_lightCNN
    if modelName == "facenet":
        from facenet_tf import getRep_facenet_tf
        getRep = getRep_facenet_tf

    dirlist = sorted(os.listdir(webfaceRoot))[:peopleNum]

    for index, dir in enumerate(dirlist):
        print("%d people running..." %index)
        for root, d , files in os.walk(webfaceRoot+dir):
            for file in files[:]:
                imgPath = webfaceRoot+dir+"/"+file
                #print(imgPath)
                img = cv2.imread(imgPath)
                try:
                    rep = getRep(img)
                except:
                    continue
                datax.append(rep)
                datay.append(index)
    datax = np.array(datax)
    datay = np.array(datay)

    f = h5py.File(saveFilePath, 'w')
    f['data'] = datax
    f['labels'] = datay
    f.close()

# 加载某种模型的特征向量集
def loadWebfaceVec(filename):

    f = h5py.File(filename, 'r')  # 打开h5文件
    print(f.keys())  # 可以查看所有的主键
    datax = f['data'][:]  # 取出主键为data的所有的键值
    datay = f["labels"][:]
    print(datax.shape)
    print(datay.shape)
    f.close()

    return datax, datay


# 使用webface数据集生成正负样本对
def createPairsWebface(saveFileName):

    posNum = 2
    negNum = 2
    imgSize = 128

    trainPair1 = []
    trainPair2 = []
    label = []

    peopleList = sorted(os.listdir(webfaceRoot)) # 获取所有人名列表

    for index, dir in enumerate(peopleList):
        print("%d people running..." % index)

        # 选取anchor样本
        posImgList = os.listdir(webfaceRoot + dir)
        anchorIndex = np.random.randint(0, len(posImgList))
        anchor = posImgList[anchorIndex]

        anchorImgPath = webfaceRoot + dir + "/" + anchor
        print("anchor img path:", anchorImgPath)
        anchorImg = cv2.imread(anchorImgPath)

        try:
            anchorImg = findAlignFace_dlib(anchorImg, imgSize)
        except:
            continue
        anchorImg = cv2.cvtColor(anchorImg, cv2.COLOR_RGB2GRAY)
        anchorImg = anchorImg[:, :, np.newaxis]

        for i in range(negNum):

            # 选取负样本人
            tmpPeopleList = copy.deepcopy(peopleList)
            del tmpPeopleList[index] # 除去本人的图片集

            # 随机选取一个人作为负样本集
            negPeople = tmpPeopleList[np.random.randint(0, len(tmpPeopleList))]
            negPeopleImgList = os.listdir(webfaceRoot+negPeople)
            # 随机选一张图片作为负样本
            neg = negPeopleImgList[np.random.randint(0, len(negPeopleImgList))]
            negImgPath = webfaceRoot+negPeople+"/"+neg
            print("neg img path:", negImgPath)
            negImg = cv2.imread(negImgPath)

            try:
                negImg = findAlignFace_dlib(negImg, imgSize)
            except:
                continue
            negImg = cv2.cvtColor(negImg, cv2.COLOR_RGB2GRAY)
            negImg = negImg[:, :, np.newaxis]


            trainPair1.append(anchorImg)
            trainPair2.append(negImg)
            label.append(-1)

        for i in range(posNum):

            # 随机选取正样本
            tmpPosImgList = copy.deepcopy(posImgList)
            del tmpPosImgList[anchorIndex] # 除去anchor图片
            pos = posImgList[np.random.randint(0, len(posImgList))]

            posImgPath = webfaceRoot + dir + "/" + pos
            print("pos img path:", posImgPath)
            posImg = cv2.imread(posImgPath)

            try:
                posImg = findAlignFace_dlib(posImg , imgSize)
            except:
                continue
            posImg = cv2.cvtColor(posImg, cv2.COLOR_RGB2GRAY)
            posImg = posImg[:, :, np.newaxis]

            trainPair1.append(anchorImg)
            trainPair2.append(posImg)
            label.append(1)

    f = h5py.File(saveFileName, 'w')
    f['train_pair1'] = np.array(trainPair1)
    f['train_pair2'] = np.array(trainPair2)
    f['label'] = np.array(label)
    f.close()
    #return np.array(trainPair1), np.array(trainPair2), np.array(label)

# 加载webface正负样本对
def loadPairsWebface(filename):

    f = h5py.File(filename, 'r')  # 打开h5文件
    trainPairs1 = f['train_pair1'][:]
    trainPairs2 = f['train_pair2'][:]
    label = f['label'][:]
    f.close()
    return trainPairs1, trainPairs2, label

# 生成webface原始数据h5文件
def createWebfaceRawData(imgSize, saveFilePath):

    peopleNum = 5000
    singleSubNum = 15
    datax = []
    datay = []
    dirlist = sorted(os.listdir(webfaceRoot))[:peopleNum]

    for index, dir in enumerate(dirlist):
        print("%d people running..." % index)
        for root, d, files in os.walk(webfaceRoot + dir):
            print("img num:", len(files))
            for file in files[:]:
                imgPath = webfaceRoot + dir + "/" + file
                img = cv2.imread(imgPath)
                try:
                    faceImg = findAlignFace_dlib(img, imgSize)
                except:
                    continue
                datax.append(faceImg)
                datay.append(index)
    datax = np.array(datax)
    datay = np.array(datay)

    f = h5py.File(saveFilePath, 'w')
    f['data'] = datax
    f['labels'] = datay
    f.close()

# 加载webface原始数据h5文件
def loadWebfaceRawData(filename):

    f = h5py.File(filename, 'r')  # 打开h5文件
    data = f['data'][:]
    label = f['labels'][:]
    f.close()
    return data, label

if __name__ == '__main__':

    #modelName = "VGGface"
    modelName = "openface"
    # modelName = "lightCNN"
    # modelName = "facenet"


    # 生成某种模型的特征向量集
    #createWebfaceVec(modelName, '/disk1/zhangxu_new/webface_vec_'+modelName+'_v2.h5')

    # 加载某种模型的特征向量集
    #loadWebfaceVec('/disk1/zhangxu_new/webface_vec_'+modelName+'_v2.h5')


    # 生成webface数据集的pair对
    # pairsFileName = '/disk1/zhangxu_new/webface_pairs_gray_v1.h5'
    # #
    # createPairsWebface(pairsFileName)
    # train1, train2, label = loadPairsWebface(pairsFileName)
    # print(train1.shape)
    # print(train2.shape)
    # print(label.shape)

    webfaceRawDataFile = '/disk1/zhangxu_new/webface_origin_data_v2.h5'
    createWebfaceRawData(imgSize=128, saveFilePath=webfaceRawDataFile)
    data, label = loadWebfaceRawData(webfaceRawDataFile)
    print(data.shape)
    print(label.shape)




