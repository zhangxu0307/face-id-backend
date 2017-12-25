from keras.layers import *
from keras.regularizers import *
import keras.backend as K
from keras.models import Model
from keras import optimizers
from keras import losses

from face_algorithm.lfw import *
from keras.applications import *
from keras_vggface.vggface import VGGFace

# 自己定义的loss
def myloss(y_true, y_pred):
    # pair1 = y_pred[0]
    # pair2 = y_pred[1]
    #loss = K.mean(-(y_true*(K.l2_normalize(pair1, axis=-1)*K.l2_normalize(pair2, axis=-1))), axis = -1)
    loss = K.mean(-(y_true * y_pred))
    return loss

# 计算余弦相似度，作为定制layer的参数
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

# 计算欧氏距离, 作为定制layer的参数
def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def l2_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

# 对比loss
def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))


# 基模型
class BaseModel():
    def __init__(self, inputSize, load, loadModelPath):
        self.inputSize = inputSize
        self.reg = 0.0
        self.model = self.buildModel()
        if load:
            self.model.load_weights(loadModelPath)

        print(self.model.summary())
        #self.opt = optimizers.Adam(lr=0.01)
        self.opt = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=False)
        self.model.compile(optimizer=self.opt, loss=contrastive_loss, metrics=['accuracy'])

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

        x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(self.reg))(totalInput)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        #x = BatchNormalization()(x)

        x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        #x = BatchNormalization()(x)

        x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        #x = BatchNormalization()(x)

        # x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(0.01))(x)
        # x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Flatten()(x)

        # x = Dense(1024, activation="relu")(x)
        # x = Dropout(0.5)(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu", kernel_regularizer=l2(0.0))(x)
        o = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input1, input2], outputs=o)

        return model

# 带预训练的孪生网络模型
class Siam_Model(BaseModel):

    def buildSubCNN(self):

        inputx = Input(shape=(self.inputSize, self.inputSize, 3))

        x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(self.reg))(inputx)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(self.reg))(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        x = BatchNormalization()(x)

        # x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu", kernel_regularizer=l2(0.01))(x)
        # x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        o = Dense(512, activation="linear")(x)

        model = Model(inputs=inputx, outputs=o)
        return model


    def buildModel(self):

        input1 = Input(shape=(self.inputSize, self.inputSize, 3))
        input2 = Input(shape=(self.inputSize, self.inputSize, 3))

        subCnn = self.buildSubCNN()

        #preTrainModel = xception.Xception(include_top=False, weights='imagenet', pooling="avg")
        #subCnn = VGGFace(include_top=False, model="resnet50", input_shape=(self.inputSize, self.inputSize, 3), pooling='avg')
        #subCnn.trainable = False

        x1 = subCnn(input1)
        x2 = subCnn(input2)

        x1 = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x1)
        x2 = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x2)
        #o = K.l2_normalize(x1 - x2, axis=-1)
        #o = Lambda(lambda x: K.dot(x[0], x[1])/K.norm, output_shape=(1,))([x1, x2])
        #o = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([x1, x2])
        o = Lambda(euclidean_distance, output_shape=l2_dist_output_shape)([x1, x2])

        model = Model(inputs=[input1, input2], outputs=o)

        return model



if __name__ == '__main__':


    #saim1 = Siam_Channel_Model(128, load=False, loadModelPath=None)
    saim2 = Siam_Model(128, load=False, loadModelPath=None)




