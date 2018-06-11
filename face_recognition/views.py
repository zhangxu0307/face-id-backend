# 使用哪一种算法就import哪一个
# from face_algorithm.face_id import getRep_openface
from face_algorithm.vgg_face import getRep_VGGface
# from face_algorithm.sphere_face_pt import getRep_SphereFace

from face_algorithm.id_utils import  calcCossimilarity, addFaceVec, deleteFaceVec
from face_algorithm.joint_bayes_face import jointBayesVerify
import cv2
import os
import pandas as pd

# django 相关, 此处只要有一个在sphereface之前import torch就会出错
from django.conf import settings
from rest_framework.response import Response
from .my_serializers import RecognitionResultSerializer, RegisterSerializer, RecognitionRequestSerializer
from .models import Info
from rest_framework.views import APIView

if settings.METHOD == "VGGface":
    getRep = getRep_VGGface
if settings.METHOD == "sphereface":
    getRep = getRep_SphereFace
# getRep = getRep_openface

# 人脸识别api
class FaceRecognition(APIView):

    def post(self, request, format=None):

        if len(settings.CANDIDATE) == 0:
            return Response({"detail":"No face in database!"})

        serializer = RecognitionRequestSerializer(data=self.request.data)

        data = serializer.valid_data
        imgArr = data["picture"]
        boundingbox = data["boundingbox"]
        threshold = data["threshold"]

        print("img:", imgArr.shape)
        print("bdbox:", boundingbox)
        print("threshold:", threshold)

        # 召回相似度最高的人
        try:
            resultId, similarity, v1, v2 = calcCossimilarity(imgArr, settings.CANDIDATE, getRep)
        except:
            return Response({"detail": "recognition failed!"})

        print("resultId:", resultId)
        print("similarity:", similarity)

        # if similarity >= settings.SIMILARITY_THRESHOLD:
        info = Info.objects.get(ID=resultId)
        ID = info.ID
        name = info.name
        resImgPath = info.imgPath
        resSerializer = RecognitionResultSerializer(resImgPath, ID, name, similarity, True)

        # 使用joint bayes进行二次验证
        jointBayesScore = jointBayesVerify(v1, v2)
        print("joint bayes score:", jointBayesScore)
        if jointBayesScore > settings.JOINT_BAYES_THRESHOLD:
            return Response(resSerializer.valid_data)
        else:
            #resSerializer = RecognitionResultSerializer(None, similarity, False)
            return Response({"detail": "no result!"})
        # else:
        #     return Response({"detail": "no result!"})

# 从相机注册api
class Register(APIView):

    def post(self, request, format=None):

        serializer = RegisterSerializer(data=self.request.data)

        data = serializer.valid_data
        imgArr = data["picture"]
        del data["picture"]
        del data["boundingbox"]
        data["imgPath"] = settings.IMAGEPATH+str(data["ID"])+".jpg"
        try:
            # 储存数据库操作
            Info.objects.create(**data)
            # 生成图片操作
            cv2.imwrite(data["imgPath"], imgArr)
            # 生成特征向量并存储
            addFaceVec(imgArr, data["ID"], getRep)
        except:
            return Response({"detail": "Database Info saved Error!"})

        return Response({"detail": "new face has been saved!"})


# 删除记录api
class DeleteFace(APIView):

    def post(self, request, format=None):

        deleteID = self.request.data["delete_ID"]
        #try:
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

        #except:

            #return Response({"detail": "delete failed!"})

# 清空记录api
class DeleteAllRecord(APIView):

    def post(self, request, format=None):

        # 删除所有的meida文件中的特征向量文件和图片
        delList = os.listdir(settings.IMAGEPATH)

        print("delete folder:", settings.IMAGEPATH)
        print("delete files list:", delList)

        for f in delList:
            filePath = os.path.join(settings.IMAGEPATH, f)
            if os.path.isfile(filePath):
                os.remove(filePath)
        print("img files clear finished!")

        # 清理当前内存中的候选特征向量
        settings.CANDIDATE = pd.DataFrame()
        print("feature vec clear finished!")

        # 清理数据库
        Info.objects.all().delete()
        print("database clear finished!")

        return Response({"detail": "all data has been cleaned!"})


# 从文件夹中直接构建数据库信息
class RegisterFromDir(APIView):

    def post(self, request, format=None):

        os.path.exists(settings.RAWFACEIMGPATH)

        files = os.listdir(settings.RAWFACEIMGPATH)

        #for (root, dirs, files) in os.walk(settings.RAWFACEIMGPATH):

        for filename in files:

            ID, name = filename.split()
            name = name.split('.')[0]
            print("ID: %s, name: %s" %(ID, name))
            imgPath = os.path.join(settings.RAWFACEIMGPATH, filename)

            imgArr = cv2.imread(imgPath)

            data = {}
            data["ID"] = ID
            data["name"] = name
            data["description"] = ""
            data["imgPath"] = settings.IMAGEPATH + str(data["ID"]) + ".jpg"

            try:
                # 生成特征向量并存储
                addFaceVec(imgArr, data["ID"], getRep)
                # 储存数据库操作
                Info.objects.create(**data)
                # 生成图片操作
                cv2.imwrite(data["imgPath"], imgArr)

            except:
                return Response({"detail": "Database Info saved Error!"})

        return Response({"detail": "all face has been saved!"})



