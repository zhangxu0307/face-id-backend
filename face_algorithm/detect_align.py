import openface
import numpy as np
import cv2
import os
import dlib

# openface\dlib模型路径
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')

# 68 landmark模型参数加载
dlibModelPath = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat") # 68特征点
align = openface.AlignDlib(dlibModelPath)

# 5 landmark模型参数加载
predictor_path = os.path.join(dlibModelDir, 'shape_predictor_5_face_landmarks.dat') # 5特征点
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 使用openface中的dlib工具检测并对齐人脸，主要对齐模型
def findAlignFace_dlib(rgbImg, imgDim):

    if rgbImg is None:
        raise Exception("Unable to load image")

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face")

    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image")

    return alignedFace

# dlib检测landmark点，dlib-5-landmark为眼角4点加鼻子1点
def findLandMarks_dlib(img):

    landmarksList = []
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {}, Part 2: {}, Part 3: {}, Part 4: {},".format(shape.part(0),
                                                  shape.part(1), shape.part(2), shape.part(3), shape.part(4)))
        for i in range(5):
            landmarksList.append(int(shape.part(i).x))
            landmarksList.append(int(shape.part(i).y))

    return landmarksList


if __name__ == '__main__':

    f = "./data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    img = cv2.imread(f)
    landMarkList = findLandMarks_dlib(img)
    print(landMarkList)



