from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import cv2
#from .serializers import RecognitionResultSerializer, RegisterSerializer, RecognitionRequestSerializer
from django.utils.six import BytesIO
from rest_framework.parsers import JSONParser
from django.http import HttpResponse
import json
import cv2
import numpy as np
from django.core.files.storage import get_storage_class
from django.core.files.base import ContentFile
from PIL import Image
import io
from face_algorithm.id_utils import  calcCossimilarity, addFaceVec, calcEuclidDistance, deleteFaceVec
from django.conf import settings
from .my_serializers import RecognitionResultSerializer, RegisterSerializer, RecognitionRequestSerializer
from .models import Info
import os

from face_algorithm.joint_bayes.joint_bayesian import Verify
from face_algorithm.joint_bayes_face import A, G
# Create your views here.


class FaceRecognition(APIView):

    def post(self, request, format=None):

        if len(settings.CANDIDATE) == 0:
            return Response({"detail":"No face in database!"})

        serializer = RecognitionRequestSerializer(data=self.request.data)

        data = serializer.valid_data
        imgArr = data["picture"]
        boundingbox = data["boundingbox"]
        threshold = data["threshold"]
        threshold = 0.8

        print("img:", imgArr.shape)
        print("bdbox:", boundingbox)
        print("threshold:", threshold)

        jointBayesThreshold = 30 # joint bayes的阈值，


        # 召回相似度最高的人
        try:
            resultId, similarity, v1, v2 = calcCossimilarity(imgArr, settings.CANDIDATE)
            #resultId, similarity = calcEuclidDistance(imgArr, settings.CANDIDATE)
        except:
            return Response({"detail": "recognition failed!"})
        print("resultId:", resultId)
        print("similarity:", similarity)
        if similarity >= threshold:
            info = Info.objects.get(ID=resultId)
            ID = info.ID
            name = info.name
            resImgPath = info.imgPath
            resSerializer = RecognitionResultSerializer(resImgPath, ID, name, similarity, True)

            # 使用joint bayes进行二次验证
            jointBayesScore = Verify(A, G, v1, v2)
            print(jointBayesScore)
            if (jointBayesScore > jointBayesThreshold):
                return Response(resSerializer.valid_data)
            else:
                #resSerializer = RecognitionResultSerializer(None, similarity, False)
                return Response({"detail": "no result!"})


class Register(APIView):

    def post(self, request, format=None):

        serializer = RegisterSerializer(data=self.request.data)

        data = serializer.valid_data
        imgArr = data["picture"]
        del data["picture"]
        del data["boundingbox"]
        data["imgPath"] = settings.IMAGEPATH+str(data["ID"])+".jpg"
        try:
            Info.objects.create(**data)
        except:
           return Response({"detail": "Database Info saved Error!"})

        cv2.imwrite(data["imgPath"], imgArr)

        # 生成特征向量并存储
        addFaceVec(imgArr, data["ID"])

        return Response({"detail": "new face has been saved!"})

class DeleteFace(APIView):

    def post(self, request, format=None):

        deleteID = self.request.data["delete_ID"]
        try:
            # 获取图片路径
            info = Info.objects.get(ID=deleteID)
            deleteImgPath = info.imgPath
            # 删除特征向量
            deleteFaceVec(deleteID)
            # 删除图片文件
            os.remove(deleteImgPath)
            # 删除数据库记录
            Info.objects.get(ID=deleteID).delete()
            return Response({"detail": "delete success!"})

        except:

            return Response({"detail": "delete failed!"})



