import cv2
#from face_algorithm.MTCNN_keras.mtcnn_caffe_weight import detectFace # 注意这里TensorFlow和pytorch不能兼容使用
from face_algorithm.MTCNN_pytorch.src import detect_faces
from PIL import Image


# # MTCNN检测face landmark点，眼睛、鼻子和嘴角5个点, keras版本
# def findLandMarks_MTCNN_keras(img):
#
#     threshold = [0.6, 0.6, 0.7]
#     maxArea = 0
#     landmarkList = []
#     rectangles = detectFace(img, threshold)
#     for rectangle in rectangles:
#         print(rectangle)
#         area = (rectangle[2]-rectangle[0])*(rectangle[3]-rectangle[1]) # 计算面积，取面积最大的face
#         if area > maxArea:
#             maxArea = area
#             for i in range(5, 15, 2):
#                 landmarkList.append(int(rectangle[i + 0]))
#                 landmarkList.append(int(rectangle[i + 1]))
#
#     return landmarkList

# MTCNN检测face landmark点，眼睛、鼻子和嘴角5个点, pytorch版本
def findLandMarks_MTCNN_pytorch(img):

    landMarkList = []

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # opencv格式转PIL Image格式

    bounding_boxes, landmarks = detect_faces(image)

    for i in range(5):                       # 整理landmark为指定形式list
        landMarkList.append(landmarks[0][i])
        landMarkList.append(landmarks[0][i+5])

    return bounding_boxes, landMarkList


if __name__ == '__main__':

    f = "./data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    img = cv2.imread(f)
    #findLandMarks_dlib(img)
    #landMarkList = findLandMarks_MTCNN_keras(img)
    landMarkList = findLandMarks_MTCNN_pytorch(img)
    print(landMarkList)