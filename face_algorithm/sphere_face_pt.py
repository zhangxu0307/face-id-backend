import cv2
import numpy as np
from face_algorithm.landmarks_mtcnn import findLandMarks_MTCNN_pytorch

import torch
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import face_algorithm.sphere_face_pytorch.net_sphere as net_sphere


from face_algorithm.sphere_face_pytorch.matlab_cp2tform import get_similarity_transform_for_cv2


# 加载sphereface_pytorch模型
net = getattr(net_sphere, 'sphere20a')()
modelPath = "./models/sphere20a_20171020.pth"
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

    # x1 = int(boundingBox[0][0])
    # y1 = int(boundingBox[0][1])
    # x2 = int(boundingBox[0][2])
    # y2 = int(boundingBox[0][3])
    # faceImg = rgbImg[y1:y2, x1:x2]

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




    # with open('./sphere_face_pytorch/data/pairs.txt') as f:
    #     pairs_lines = f.readlines()[1:]
    #
    # for i in range(6000):
    #     p = pairs_lines[i].replace('\n', '').split('\t')
    #
    #     if 3 == len(p):
    #         sameflag = 1
    #         name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    #         name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
    #     if 4 == len(p):
    #         sameflag = 0
    #         name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    #         name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
    #     img1 = cv2.imread("./data/lfw/" + name1)
    #     img2 = cv2.imread("./data/lfw/" + name2)
    #     #print(landmark[name1])
    #     # img1 = alignment(img1, landmark[name1])
    #     # img2 = alignment(img2, landmark[name2])
    #
    #     rep1 = getRep_SphereFace(img1)
    #     rep2 = getRep_SphereFace(img2)
    #     # img1 = img1.transpose(2, 0, 1).reshape((1, 3, 112, 96))
    #     # img1 = (img1 - 127.5) / 128.0
    #     # img2 = img2.transpose(2, 0, 1).reshape((1, 3, 112, 96))
    #     # img2 = (img2 - 127.5) / 128.0
    #     #
    #     # img1 = Variable(torch.from_numpy(img1).float(), volatile=True).cuda()
    #     # output1 = net(img1)
    #     # img2 = Variable(torch.from_numpy(img2).float(), volatile=True).cuda()
    #     # output2 = net(img2)
    #     # output1 = output1.
    #     #cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    #     cosdistance = np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2))
    #     print('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))





