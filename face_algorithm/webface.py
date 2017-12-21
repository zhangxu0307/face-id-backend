import numpy as np
import pandas as pd
import h5py
import os
import cv2
import copy
from face_algorithm.detect_align import findAlignFace_dlib

webfaceRoot = "/disk1/zhangxu_new/CASIA-WebFace/"


# 生成某种模型的特征向量集
def createWebfaceVec(modelName):

    peopleNum = 8000 # 选取人数
    singleSubNum = 25 # 每人选取图片数
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
            for file in files[:singleSubNum]:
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

    f = h5py.File('/disk1/zhangxu_new/webface_vec_'+modelName+'.h5', 'w')
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
def createPairsWebface():

    posNum = 5
    negNum = 5
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
                negImg = findAlignFace_dlib(negImg , imgSize)
            except:
                continue

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

            trainPair1.append(anchorImg)
            trainPair2.append(posImg)
            label.append(1)

    f = h5py.File('/disk1/zhangxu_new/webface_pairs_v2.h5', 'w')
    f['train_pair1'] = np.array(trainPair1)
    f['train_pair2'] = np.array(trainPair2)
    f['label'] = np.array(label)
    f.close()
    #return np.array(trainPair1), np.array(trainPair2), np.array(label)

# 加载webface正负样本对
def loadPairsWebface():

    f = h5py.File('/disk1/zhangxu_new/webface_pairs.h5', 'r')  # 打开h5文件
    trainPairs1 = f['train_pair1'][:]
    trainPairs2 = f['train_pair2'][:]
    label = f['label'][:]
    f.close()
    return trainPairs1, trainPairs2, label

if __name__ == '__main__':

    #modelName = "VGGface"
    #modelName = "openface"
    # modelName = "lightCNN"
    # modelName = "facenet"

    # 生成某种模型的特征向量集
    #createWebfaceVec(modelName)

    # 加载某种模型的特征向量集
    #loadWebfaceVec('/disk1/zhangxu_new/webface_vec_'+modelName+'.h5')

    createPairsWebface()
    train1, train2, label = loadPairsWebface()
    print(train1.shape)
    print(train2.shape)
    print(label.shape)



