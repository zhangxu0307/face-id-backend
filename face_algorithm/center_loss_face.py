from face_algorithm.center_loss.model import *
from face_algorithm.webface import *
from keras.utils.np_utils import to_categorical
from face_algorithm.lfw import *
from face_algorithm.detect_align import findAlignFace_dlib


# 测试centern loss cnn LFW数据集相似度分布情况
def runLFW(model, trainLabel, imgSize):

    acc = 0
    posScore = []
    negScore = []

    count = 0
    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        count += 1
        if count > 10:
            break
        try:
            faceImg1 = findAlignFace_dlib(img1, imgSize)
            faceImg2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        faceImg1 = (faceImg1[np.newaxis, :, :, :]-127.5)/128.0
        faceImg2 = (faceImg2[np.newaxis, :, :, :]-127.5)/128.0
        rep1 = model.getRepVec(faceImg1, trainLabel)
        rep2 = model.getRepVec(faceImg2, trainLabel)
        score = calcCosSimilarityPairs(rep1, rep2)
        print(score)
        #posScore.append(score)

    #posCsv = pd.DataFrame(posScore)
    #posCsv.to_csv("./data/pos_score_"+modelName+".csv", index=False)

    count = 0
    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        count += 1
        if count > 10:
            break
        try:
            faceImg1 = findAlignFace_dlib(img1, imgSize)
            faceImg2 = findAlignFace_dlib(img2, imgSize)
        except:
            continue
        faceImg1 = (faceImg1[np.newaxis, :, :, :] - 127.5) / 128.0
        faceImg2 = (faceImg2[np.newaxis, :, :, :] - 127.5) / 128.0
        rep1 = model.getRepVec(faceImg1, trainLabel)
        rep2 = model.getRepVec(faceImg2, trainLabel)
        score = calcCosSimilarityPairs(rep1, rep2)
        print(score)
        #negScore.append(score)

    #negCsv = pd.DataFrame(negScore)
    #negCsv.to_csv("./data/neg_score_"+modelName+".csv", index=False)

if __name__ == '__main__':

    imgSize = 128
    classNum = 10575
    webfaceRawDataFile = '/disk1/zhangxu_new/webface_origin_data.h5'
    modelPath = "./models/center_loss_cnn_v2.h5"
    data, label = loadWebfaceRawData(webfaceRawDataFile)
    model = CenterLossModel(inputSize=128, classNum=classNum, dim=512, lamda=0.001, load=True,
                                      loadModelPath=modelPath)
    runLFW(model, label, imgSize)

