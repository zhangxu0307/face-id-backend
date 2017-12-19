from keras.layers import *
import keras
from keras.models import Model
from keras import optimizers
from keras import losses

class Siam_Channel_Model():

    def __init__(self, inputSize):
        self.inputSize = inputSize
        self.model = self.buildModel()
        print(self.model.summary())
        self.opt = optimizers.Adam(lr=0.1)
        self.model.compile(optimizer=self.opt, loss=losses.hinge, metrics=None,)

    def buildModel(self):

        input1 = Input(shape=(self.inputSize, self.inputSize, 3))
        input2 = Input(shape=(self.inputSize, self.inputSize, 3))

        totalInput = concatenate([input1, input2], axis=3)

        x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu")(totalInput)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Flatten()(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.7)(x)
        o = Dense(1)(x)

        model = Model(inputs=[input1, input2], outputs=o)

        return model

    def train(self, trainPair1, trainPair2, trainy, epoch, batchSize):
        self.model.fit([trainPair1, trainPair2], trainy, epochs=epoch,
                       batch_size=batchSize, validation_split=0.1)

    def inference(self, testPair1, testPair2):
        res = self.model.predict([testPair1, testPair2])
        return res

if __name__ == '__main__':

    saim1 = Siam_Channel_Model(96)




