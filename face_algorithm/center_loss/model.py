
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras import losses
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras import initializers
from keras import backend as K
from face_algorithm.webface import *
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# center loss 定制层
class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        classNum = 100
        dim = 512
        self.centers = self.add_weight(name='centers',
                                       shape=(classNum, dim),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nxdim, 为特征向量
        # x[1] is Nxclassnum onehot, 为label
        # self.centers is classnumxdim, 为类中心向量
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)

class CenterLossModel():

    def __init__(self, inputSize, classNum, lamda):

        self.inputSize = inputSize
        self.reg = 0.0
        self.activity = "relu"
        self.classNum = classNum
        self.model = self.buildModel()
        print(self.model.summary())
        
        #opt = optimizers.Adam(0.1)
        opt = optimizers.SGD(lr=0.1, momentum=0.9)
        self.model.compile(optimizer=opt,
                      loss=[losses.categorical_crossentropy, zero_loss],
                      loss_weights=[1, lamda], metrics=['accuracy'])


    def buildModel(self):

        inputx = Input(shape=(self.inputSize, self.inputSize, 3))
        inputy = Input(shape=(self.classNum,))

        x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="SAME", activation=self.activity,
                   kernel_regularizer=l2(self.reg))(inputx)
        # x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="SAME", activation=self.activity,
        #            kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        #x = BatchNormalization()(x)

        x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation=self.activity,
                   kernel_regularizer=l2(self.reg))(x)
        # x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation=self.activity,
        #            kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        #x = BatchNormalization()(x)

        x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation=self.activity,
                   kernel_regularizer=l2(self.reg))(x)
        # x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation=self.activity,
        #            kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        #x = BatchNormalization()(x)

        # x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu",
        #            kernel_regularizer=l2(self.reg))(x)
        # x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        # x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(512, activation=self.activity)(x)

        mainOutput = Dense(self.classNum, activation='softmax', name='main_out')(x)
        sideOutput = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, inputy])

        # mainModel = Model(inputs=inputx, outputs=mainOutput)
        # sideModel = Model(inputs=[inputx, inputy], outputs=sideOutput)
        model = Model(inputs=[inputx, inputy], outputs=[mainOutput, sideOutput])

        return model

    def train(self, trainx, trainy, epoch, batchSize):
        self.model.fit([trainx, trainy], [trainy, trainy], epochs=epoch,
                       batch_size=batchSize, validation_split=0.1)

    def inference(self, test):
        res = self.model.predict(test)
        return res

    def saveModel(self, modelPath):
        self.model.save_weights(modelPath)

if __name__ == '__main__':

    webfaceRawDataFile = '/disk1/zhangxu_new/webface_origin_data.h5'
    modelPath = "../models/center_loss_cnn.h5"
    data, label = loadWebfaceRawData(webfaceRawDataFile)

    label = to_categorical(label)
    print(label.shape)

    centerLossModel = CenterLossModel(inputSize=128, classNum=100, lamda=0.1)
    centerLossModel.train(data, label, epoch=5, batchSize=256)
    centerLossModel.saveModel(modelPath)

