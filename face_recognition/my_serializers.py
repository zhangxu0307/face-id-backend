import base64
import json
from PIL import Image
import io
import numpy as np
import cv2


class Result(object):
    def __init__(self, img, similarity, flag):
        self.picture = img
        self.similarity = similarity
        self.flag = flag


# 识别结果返回序列化器
class RecognitionResultSerializer():

    def __init__(self, imgPath, ID, name, similarity, flag):

        self.imgPath = imgPath
        self.ID = ID
        self.name = name
        self.similarity = similarity
        self.flag = flag
        self._encode()

    def _encode(self):


        fin = open(self.imgPath, 'rb')
        image_data = fin.read()
        base64_data = base64.b64encode(image_data)
        self.valid_data = {'picture': base64_data.decode(), 'similarity': self.similarity,
                 'detail': "find result!", 'ID':self.ID, "name":self.name}

        #此处不需要json序列化，response已经完成此任务



# 客户端请求序列化器
class RecognitionRequestSerializer():

    def __init__(self, data): # 此处传入的就是已经从json解析来的字典数据

        self.data = data
        self._decode()

    def _decode(self):

        ori_image_data = base64.b64decode(self.data['picture'])
        image = Image.open(io.BytesIO(ori_image_data))
        image = np.asarray(image)
        imageArr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.valid_data = self.data
        self.valid_data['picture'] = imageArr


# 客户端注册序列化器
class RegisterSerializer():

    def __init__(self, data):

        self.data = data
        self._decode()

    def _decode(self):

        ori_image_data = base64.b64decode(self.data['picture'])
        image = Image.open(io.BytesIO(ori_image_data))
        image = np.asarray(image)
        imageArr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.valid_data = self.data
        self.valid_data['picture'] = imageArr

