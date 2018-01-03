from face_algorithm.joint_bayes.joint_bayesian import *
import pickle
from django.conf import settings
import os

# 加载joint bayes模型参数
vecName = "VGGface" # joint bayes 支持的特征向量
modelPath = os.path.join(settings.BASE_DIR+"/face_algorithm/models/joint_bayes", vecName) # django工程路径写法
A_path = os.path.join(modelPath, "A.pkl")
G_path = os.path.join(modelPath, "G.pkl")
with open(A_path, "rb") as f:
    A = pickle.load(f)
with open(G_path, "rb") as f:
    G = pickle.load(f)

def jointBayesVerify(v1, v2):
    jointBayesScore = Verify(A, G, v1, v2)
    return jointBayesScore


if __name__ == "__main__":

    pass