import openface
import numpy as np
import cv2
from face_algorithm.detect_align import findAlignFace_dlib # 此处包内调用有问题
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# openface参数及模型加载
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
openfaceModelDir = os.path.join(modelDir, 'openface')
openfaceModelPath = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=True) # 输入大小为96*96

# openface模型获取人脸表示向量
def getRep_openface(rgbImg):

    alignedFace = findAlignFace_dlib(rgbImg, 96)


    rep = net.forward(alignedFace)

    return rep

if __name__ == '__main__':


    imgPath1 = "../test_json/1.jpg"
    imgPath2 = "../test_json/2.jpg"
    imgPath3 = "../test_json/3.jpg"
    imgPath4 = "../test_json/4.jpg"

    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)
    img3 = cv2.imread(imgPath3)
    img4 = cv2.imread(imgPath4)

    rep1 = getRep_openface(img1)
    rep2 = getRep_openface(img2)
    rep3 = getRep_openface(img3)
    rep4 = getRep_openface(img4)


    # 余弦相似度
    print(np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2)))
    print(np.dot(rep3, rep4.T) / (np.linalg.norm(rep3, 2) * np.linalg.norm(rep4, 2)))
    print(np.dot(rep1, rep3.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep3, 2)))
    print(np.dot(rep2, rep4.T) / (np.linalg.norm(rep2, 2) * np.linalg.norm(rep4, 2)))










