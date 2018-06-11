import torch
from torch.autograd import Variable
import face_algorithm.sphere_face_pytorch.net_sphere as net_sphere
from face_algorithm.sphere_face_pytorch.matlab_cp2tform import get_similarity_transform_for_cv2

import numpy as np
from face_algorithm.landmarks_mtcnn import findLandMarks_MTCNN_pytorch
import os

import cv2

# 加载sphereface_pytorch模型
net = getattr(net_sphere, 'sphere20a')()
# django 路径写法
modelPath = "../face_algorithm/models/sphereface/sphere20a_20171020.pth"
# 单元测试py文件测试路径写法
# modelPath = "face_algorithm/models/sphereface/sphere20a_20171020.pth"

net.load_state_dict(torch.load(modelPath))
net.cuda()
net.eval()
net.feature = True


# 图像对齐
def alignment(src_img,src_pts):

    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)

    return face_img


def getRep_SphereFace(rgbImg):

    boundingBox, landmarksList = findLandMarks_MTCNN_pytorch(rgbImg)

    alignedFace = alignment(rgbImg, landmarksList)

    alignedFace = alignedFace.astype(np.float64)
    alignedFace = (alignedFace - 127.5) / 128.0
    alignedFace = np.transpose(alignedFace, [2, 0, 1])
    x = np.expand_dims(alignedFace, axis=0)

    x = Variable(torch.from_numpy(x).float(), volatile=True).cuda()

    rep = net(x)
    rep = rep.data.cpu().numpy()
    rep = np.reshape(rep, (512,))
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

    rep1 = getRep_SphereFace(img1)
    rep2 = getRep_SphereFace(img2)
    rep3 = getRep_SphereFace(img3)
    rep4 = getRep_SphereFace(img4)

    # 余弦相似度
    print(np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2)))
    print(np.dot(rep3, rep4.T) / (np.linalg.norm(rep3, 2) * np.linalg.norm(rep4, 2)))
    print(np.dot(rep1, rep3.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep3, 2)))
    print(np.dot(rep2, rep4.T) / (np.linalg.norm(rep2, 2) * np.linalg.norm(rep4, 2)))

