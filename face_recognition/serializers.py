

from rest_framework import serializers
from drf_base64.fields import Base64ImageField
from drf_base64.serializers import ModelSerializer

class Result(object):
    def __init__(self, img, similarity, flag):
        self.picture = img
        self.similarity = similarity
        self.flag = flag


# 识别结果返回序列化器
class RecognitionResultSerializer(serializers.Serializer):

    picture = Base64ImageField(required=False)
    similarity = serializers.FloatField(read_only=True, )
    flag = serializers.IntegerField()

# 目标框序列化器
class BoundingBox(serializers.Serializer):

    x = serializers.IntegerField()
    y = serializers.IntegerField()
    w = serializers.IntegerField()
    h = serializers.IntegerField()

# 客户端请求序列化器
class RecognitionRequestSerializer(serializers.Serializer):

    picture = Base64ImageField(required=False)
    threshold = serializers.FloatField()
    boundingbox = BoundingBox()

# 客户端注册序列化器
class RegisterSerializer(serializers.Serializer):

    picture = Base64ImageField(required=False)
    boundingbox = BoundingBox()


