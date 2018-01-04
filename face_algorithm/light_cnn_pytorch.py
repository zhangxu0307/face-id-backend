from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2

from face_algorithm.detect_align import findAlignFace_dlib

from face_algorithm.light_CNN_pytorch.light_cnn import LightCNN_9Layers, LightCNN_29Layers
modelPath = "./models/lightCNN_pytorch/LightCNN_29Layers_checkpoint.pth"
model = LightCNN_29Layers(num_classes=79077)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(modelPath)
model.load_state_dict(checkpoint['state_dict'])

transform = transforms.Compose([transforms.ToTensor()])

def gerRep_lightCNN_pytorch(rgbImg):

    faceImg = findAlignFace_dlib(rgbImg, 128)

    inputImg = torch.zeros(1, 1, 128, 128)

    grayImg = cv2.cvtColor(faceImg, cv2.COLOR_RGB2GRAY)
    img = np.reshape(grayImg, (128, 128, 1))
    img = transform(img)
    inputImg[0, :, :, :] = img
    input = inputImg.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    _, features = model(input_var)

    rep = features.data.cpu().numpy()
    rep = np.reshape(rep, (256,))

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

    rep1 = gerRep_lightCNN_pytorch(img1)
    rep2 = gerRep_lightCNN_pytorch(img2)
    rep3 = gerRep_lightCNN_pytorch(img3)
    rep4 = gerRep_lightCNN_pytorch(img4)

    # 余弦相似度
    print(np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2)))
    print(np.dot(rep3, rep4.T) / (np.linalg.norm(rep3, 2) * np.linalg.norm(rep4, 2)))
    print(np.dot(rep1, rep3.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep3, 2)))
    print(np.dot(rep2, rep4.T) / (np.linalg.norm(rep2, 2) * np.linalg.norm(rep4, 2)))



