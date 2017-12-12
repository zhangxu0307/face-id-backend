import openface
import numpy as np
import cv2
import os
import tensorflow as tf
from scipy import misc
#from face_algorithm.facenet.src.align import detect_face

# openface参数及模型加载
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')

dlibModelPath = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

align = openface.AlignDlib(dlibModelPath)

# 使用openface中的dlib工具检测并对齐人脸
def findAlignFace_dlib(rgbImg, imgDim):

    if rgbImg is None:
        raise Exception("Unable to load image")

    rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB) # 转换RGB

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face")

    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image")

    return alignedFace


# class Detection:
#
#     # face detection parameters
#     minsize = 20  # minimum size of face
#     threshold = [0.6, 0.7, 0.7]  # three steps's threshold
#     factor = 0.709  # scale factor
#
#     def __init__(self, face_crop_size=160, face_crop_margin=32):
#         self.pnet, self.rnet, self.onet = self._setup_mtcnn()
#         self.face_crop_size = face_crop_size
#         self.face_crop_margin = face_crop_margin
#
#     def _setup_mtcnn(self):
#         with tf.Graph().as_default():
#             gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#             sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#             with sess.as_default():
#                 return align.detect_face.create_mtcnn(sess, None)
#
#     def find_largest_faces(self, image):
#         #faces = []
#
#         bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
#                                                           self.pnet, self.rnet, self.onet,
#                                                           self.threshold, self.factor)
#         areaMax = 0
#         for bb in bounding_boxes:
#
#             bounding_box = np.zeros(4, dtype=np.int32)
#
#             img_size = np.asarray(image.shape)[0:2]
#             bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
#             bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
#             bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
#             bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
#             area = np.abs(bounding_box[3]-bounding_box[1])*np.abs(bounding_box[2]-bounding_box[0])
#             print(area)
#             if area > areaMax:
#                 areaMax = area
#                 cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
#                 faceimg = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
#
# #            faces.append(bounding_box)
#
#         return faceimg
