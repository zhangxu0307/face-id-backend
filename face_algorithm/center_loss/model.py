
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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
classNum = 10575
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
                                       shape=(classNum, 512),
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

        opt = optimizers.Adam(1e-3)
        #opt = optimizers.SGD(lr=1e-2, momentum=0.9)
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
        x = BatchNormalization()(x)

        #第二层
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weight_decay))(
            x)
        x = prelu(x)
        # x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weight_decay))(
        #     x)
        # x = prelu(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = BatchNormalization()(x)


        #第三层
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_regularizer=l2(self.weight_decay))(x)
        x = prelu(x)
        # x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
        #            kernel_regularizer=l2(self.weight_decay))(x)
        # x = prelu(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = BatchNormalization()(x)
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

    # 图片生成器，采用keras的工具实现
    def createImgGenerator(self, trainFilePath, batchSize):

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            # featurewise_center=True,
            shear_range=0.2,
            zoom_range=0.2,
            # samplewise_std_normalization=False,
            # zca_whitening=False,
            rotation_range=5,
            width_shift_range=0.0,
            height_shift_range=0.0,
            channel_shift_range=0.0,
            fill_mode='nearest',
            # cval=0.,
            # preprocessing_function=preprocessing_img,
            # preprocessing_function=PCA_Jittering,
            vertical_flip=False,
            horizontal_flip=True
        )

        #test_datagen = ImageDataGenerator(rescale=1. / 255)

        baseTrainGenerator = train_datagen.flow_from_directory(
            trainFilePath,
            target_size=(self.inputSize, self.inputSize),
            batch_size=batchSize, shuffle=True,
            class_mode="categorical")  # categorical返回one-hot的类别，binary返回单值

        return baseTrainGenerator

    # 训练数据生成器，上一个函数的wrapper，实现生成训练所需的数据格式
    def createTrainDataenerator(self, trainFilePath, batchSize):

        baseTrainGenerator = self.createImgGenerator(trainFilePath, batchSize)
        while True:
            datax, datay = baseTrainGenerator.next()
            dummy = np.zeros((datax.shape[0], 1))
            yield [datax, datay], [datay, dummy]

    def train(self, trainx, trainy, epoch, batchSize):

        dummy1 = np.zeros((trainx.shape[0], 1))

        self.model.fit([trainx, trainy], [trainy, dummy1], epochs=epoch,
                       batch_size=batchSize)

    # 在线生成数据并做数据增强后训练
    def train_online(self, trainRootDir, steps_per_epoch, epoch, batchSize):

        self.trainGen = self.createTrainDataenerator(trainRootDir, batchSize)
        # adjustLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0, mode='auto',
        #                              epsilon=0.001, cooldown=0, min_lr=5e-6)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(self.trainGen, steps_per_epoch=steps_per_epoch, epochs=epoch,
                                 # validation_data=validation_generator,
                                 # validation_steps=10000 // batchSize,
                                 verbose=1,
                                 #callbacks=[adjustLR]
                                 )

    # 预测分类结果
    def inference(self, test):
        res = self.model.predict(test)
        return res

    # 中间层预测出特征向量
    def getRepVec(self, test, trainy):
        dummy2 = np.zeros((trainy.shape[0], self.classNum))
        base_model = self.model
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('side_out').output)
        rep = model.predict([test, dummy2])
        return rep

    #保存模型
    def saveModel(self, modelPath):
        self.model.save_weights(modelPath)



if __name__ == '__main__':

    webfaceRawDataFile = '/disk1/zhangxu_new/webface_origin_data_v2.h5'
    webfaceRootDir = "/disk1/zhangxu_new/CASIA-WebFace/"
    modelPath = "../models/center_loss_cnn_v2.h5"
    # x_train, y_train = loadWebfaceRawData(webfaceRawDataFile)
    # y_train_onehot = to_categorical(y_train)
    # classNum = y_train_onehot.shape[1]
    # print("class num:", classNum)
    #
    # x_train = (x_train-127.5)/128.0



    centerLossModel = CenterLossModel(inputSize=128, classNum=classNum, dim=512, lamda=0.001)
    #centerLossModel = CenterLossModel(inputSize=128, classNum=classNum, dim=512, lamda=0.001, load=True, loadModelPath=modelPath)
    #centerLossModel.train(x_train, y_train_onehot, epoch=10, batchSize=512)
    centerLossModel.train_online(webfaceRootDir, steps_per_epoch=100, epoch=10, batchSize=256)
    centerLossModel.saveModel(modelPath)

