
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
from keras.datasets import mnist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

### prelu

def prelu(x, name='default'):
    if name == 'default':
        return PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    else:
        return PReLU(alpha_initializer=initializers.Constant(value=0.25), name=name)(x)


### special layer

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(100, 512),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nxdim,
        # x[1] is Nxcalssnum onehot,
        # self.centers is classnum*dim
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


### custom loss

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)




class CenterLossModel():

    def __init__(self, inputSize, classNum, dim, lamda, load=False, loadModelPath=None):

        self.inputSize = inputSize
        self.reg = 0.0
        self.dim = dim
        self.weight_decay = 0.0005
        #self.activity = PReLU(alpha_initializer=initializers.Constant(value=0.25))
        #self.activity = "relu"
        self.classNum = classNum
        self.model = self.buildModel()
        if load:
            self.model.load_weights(loadModelPath)
        print(self.model.summary())

        #opt = optimizers.Adam(1e-2)
        opt = optimizers.SGD(lr=1e-3, momentum=0.9)
        self.model.compile(optimizer=opt,
                      loss=[losses.categorical_crossentropy, zero_loss],
                      loss_weights=[1, lamda], metrics=['accuracy'])


    def buildModel(self):

        inputx = Input(shape=(self.inputSize, self.inputSize, 3))
        inputy = Input(shape=(self.classNum,))

        x = BatchNormalization()(inputx)

        # 第一层
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weight_decay))(
            x)
        x = prelu(x)
        # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weight_decay))(
        #     x)
        # x = prelu(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        #第二层
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weight_decay))(
            x)
        x = prelu(x)
        # x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weight_decay))(
        #     x)
        # x = prelu(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)


        #第三层
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_regularizer=l2(self.weight_decay))(x)
        x = prelu(x)
        # x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
        #            kernel_regularizer=l2(self.weight_decay))(x)
        # x = prelu(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        #

        # 展开
        x = Flatten()(x)
        x = Dense(self.dim, kernel_regularizer=l2(self.weight_decay))(x)
        x = prelu(x, name='side_out')
        #
        main = Dense(self.classNum, activation='softmax', name='main_out', kernel_regularizer=l2(self.weight_decay))(x)
        side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, inputy])

        model = Model(inputs=[inputx, inputy], outputs=[main, side])

        return model

    def train(self, trainx, trainy, epoch, batchSize):
        dummy1 = np.zeros((trainx.shape[0], 1))

        self.model.fit([trainx, trainy], [trainy, dummy1], epochs=epoch,
                       batch_size=batchSize)

    def inference(self, test):
        res = self.model.predict(test)
        return res

    def getRepVec(self, test, trainy):
        dummy2 = np.zeros((trainy.shape[0], self.classNum))
        base_model = self.model
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('side_out').output)
        rep = model.predict([test, dummy2])
        return rep

    def saveModel(self, modelPath):
        self.model.save_weights(modelPath)



if __name__ == '__main__':

    webfaceRawDataFile = '/disk1/zhangxu_new/webface_origin_data_v3.h5'
    modelPath = "../models/center_loss_cnn.h5"
    x_train, y_train = loadWebfaceRawData(webfaceRawDataFile)
    y_train_onehot = to_categorical(y_train)

    x_train = (x_train-127.5)/128.0

    #centerLossModel = CenterLossModel(inputSize=128, classNum=100, dim=512, lamda=0.003)
    centerLossModel = CenterLossModel(inputSize=128, classNum=100, dim=512, lamda=0.003, load=True, loadModelPath=modelPath)
    centerLossModel.train(x_train, y_train_onehot, epoch=40, batchSize=256)
    centerLossModel.saveModel(modelPath)

