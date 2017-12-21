from keras.layers import *
from keras.regularizers import *
import keras
from keras.models import Model
from keras import optimizers
from keras import losses
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from face_algorithm.lfw import *
from keras.applications import *
from keras_vggface.vggface import VGGFace

# 基模型
class BaseModel():
    def __init__(self, inputSize, load, loadModelPath):
        self.inputSize = inputSize
        self.model = self.buildModel()
        if load:
            self.model.load_weights(loadModelPath)

        print(self.model.summary())
        # self.opt = optimizers.Adam(lr=0.05)
        self.opt = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=self.opt, loss=losses.hinge, metrics=['accuracy'])

    def buildModel(self):
        pass

    def train(self, trainPair1, trainPair2, trainy, epoch, batchSize):
        self.model.fit([trainPair1, trainPair2], trainy, epochs=epoch,
                       batch_size=batchSize, validation_split=0.1)

    def inference(self, testPair1, testPair2):
        res = self.model.predict([testPair1, testPair2])
        return res

    def saveModel(self, modelPath):
        self.model.save_weights(modelPath)


# 双通道孪生网络模型
class Siam_Channel_Model(BaseModel):


    def buildModel(self):

        input1 = Input(shape=(self.inputSize, self.inputSize, 3))
        input2 = Input(shape=(self.inputSize, self.inputSize, 3))

        totalInput = concatenate([input1, input2], axis=3)

        x = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(0.01))(totalInput)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(0.01))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        # x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(0.01))(x)
        # x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        # x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(0.01))(x)
        # x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Flatten()(x)

        # x = Dense(1024, activation="relu")(x)
        # x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        o = Dense(1, activation="linear")(x)

        model = Model(inputs=[input1, input2], outputs=o)

        return model

# 带预训练的孪生网络模型
class Siam_Model(BaseModel):

    def buildModel(self):

        input1 = Input(shape=(self.inputSize, self.inputSize, 3))
        input2 = Input(shape=(self.inputSize, self.inputSize, 3))

        preTrainModel = xception.Xception(include_top=False, weights='imagenet', pooling="avg")
        #preTrainModel = VGGFace(include_top=False, model="resnet50", input_shape=(128, 128, 3), pooling='avg')

        preTrainModel.trainable = False

        x1 = preTrainModel(input1)
        x2 = preTrainModel(input2)

        x1 = Dense(512)(x1)
        x2 = Dense(512)(x2)

        x =  concatenate([x1, x2])

        o = Dense(1)(x)

        model = Model(inputs=[input1, input2], outputs=o)

        return model






if __name__ == '__main__':


    saim1 = Siam_Channel_Model(128, load=False, loadModelPath=None)




