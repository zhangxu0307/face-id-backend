import numpy as np
import tensorflow as tf
from facenet.src import align
from scipy import misc
from facenet.src import facenet
import os
import cv2

from detect_align import findAlignFace_dlib

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_memory_fraction = 0.3
facenet_model_checkpoint = "facenet/model/20170511-185253"
classifier_model = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_1.pkl"
debug = False



class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, img):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(img)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

encoder = Encoder()

def getRep_facenet_tf(img, ):

    # alignedFace = dector.find_largest_faces(img)
    # print(alignedFace.shape)
    alignedFace = findAlignFace_dlib(img, 160)

    rep = encoder.generate_embedding(alignedFace)
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

    rep1 = getRep_facenet_tf(img1)
    rep2 = getRep_facenet_tf(img2)
    rep3 = getRep_facenet_tf(img3)
    rep4 = getRep_facenet_tf(img4)

    # 余弦相似度
    print(np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2)))
    print(np.dot(rep3, rep4.T) / (np.linalg.norm(rep3, 2) * np.linalg.norm(rep4, 2)))
    print(np.dot(rep1, rep3.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep3, 2)))
    print(np.dot(rep2, rep4.T) / (np.linalg.norm(rep2, 2) * np.linalg.norm(rep4, 2)))






