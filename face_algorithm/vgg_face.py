from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface import utils
import cv2
import os
from face_algorithm.detect_align import findAlignFace_dlib # 使用dilib检测和对齐


# vgg-face模型加载
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = VGGFace(include_top=False, model="resnet50", input_shape=(224, 224, 3), pooling='avg')
print(model.predict(np.zeros((1, 224, 224, 3)))) # 此处必须预先预测一次，否则django调用会报错。。。
print(model.summary())

# vggface模型回去人脸表示向量
def getRep_VGGface(rgbImg, version=2):

    alignedFace = findAlignFace_dlib(rgbImg, 224)

    alignedFace = alignedFace.astype(np.float64)

    x = np.expand_dims(alignedFace, axis=0)
    x = utils.preprocess_input(x, version=version)  # or version=2
    rep = model.predict(x)
    rep = np.reshape(rep, (2048,))
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

    rep1 = getRep_VGGface(img1)
    rep2 = getRep_VGGface(img2)
    rep3 = getRep_VGGface(img3)
    rep4 = getRep_VGGface(img4)

    # 余弦相似度
    print(np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2)))
    print(np.dot(rep3, rep4.T) / (np.linalg.norm(rep3, 2) * np.linalg.norm(rep4, 2)))
    print(np.dot(rep1, rep3.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep3, 2)))
    print(np.dot(rep2, rep4.T) / (np.linalg.norm(rep2, 2) * np.linalg.norm(rep4, 2)))




