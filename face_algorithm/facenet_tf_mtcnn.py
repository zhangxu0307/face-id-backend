import numpy as np
import tensorflow as tf
from face_algorithm.facenet.src import align
from scipy import misc

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:

            bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
            image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(bounding_box)

        return faces


