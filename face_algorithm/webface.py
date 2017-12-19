import numpy as np
import pandas as pd
import h5py
import os
import cv2

webfaceRoot = "/disk1/zhangxu_new/CASIA-WebFace/"

def createWebfaceVec(modelName):

    peopleNum = 1000 # 选取人数
    singleSubNum = 10 # 每人选取图片数
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

def loadWebfaceVec(filename):

    f = h5py.File(filename, 'r')  # 打开h5文件
    print(f.keys())  # 可以查看所有的主键
    datax = f['data'][:]  # 取出主键为data的所有的键值
    datay = f["labels"][:]
    print(datax.shape)
    print(datay.shape)
    f.close()


if __name__ == '__main__':

    #modelName = "VGGface"
    modelName = "openface"
    # modelName = "lightCNN"
    # modelName = "facenet"
    createWebfaceVec(modelName)
    loadWebfaceVec('/disk1/zhangxu_new/webface_vec_'+modelName+'.h5')


