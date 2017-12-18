from light_CNN import LCNN29
import tensorflow as tf
import numpy as np
import cv2
import scipy.io as sio
import random
from detect_align import findAlignFace_dlib


def getRep_lightCNN(imgArr):

    imgArr = findAlignFace_dlib(imgArr, 128)
    #imgArr = cv2.resize(imgArr, (144, 144))
    # w = 8
    # h = 8
    # img = imgArr[w:w + 128, h:h + 128] / 255.
    img = imgArr/255.
    img = np.float32(img)

    img = img[np.newaxis, :]
    print(img.shape)

    imgs = np.array(img)
    feas = LCNN29.eval(imgs)
    feas = np.reshape(feas, (512,))

    return feas

if __name__ == '__main__':

    imgPath1 = "../test_json/1.jpg"
    imgPath2 = "../test_json/2.jpg"
    imgPath3 = "../test_json/3.jpg"
    imgPath4 = "../test_json/4.jpg"

    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)
    img3 = cv2.imread(imgPath3)
    img4 = cv2.imread(imgPath4)

    rep1 = getRep_lightCNN(img1)
    print(rep1.shape)
    rep2 = getRep_lightCNN(img2)
    rep3 = getRep_lightCNN(img3)
    rep4 = getRep_lightCNN(img4)

    # 余弦相似度
    print(np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2)))
    print(np.dot(rep3, rep4.T) / (np.linalg.norm(rep3, 2) * np.linalg.norm(rep4, 2)))
    print(np.dot(rep1, rep3.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep3, 2)))
    print(np.dot(rep2, rep4.T) / (np.linalg.norm(rep2, 2) * np.linalg.norm(rep4, 2)))




