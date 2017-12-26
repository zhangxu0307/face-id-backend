from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras import losses
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras import initializers
from keras import backend as K

import utils
import my_callbacks
import numpy as np

### parameters

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50
weight_decay = 0.0005


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
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
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


### model

def my_model(x, labels):
    x = BatchNormalization()(x)
    #
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Flatten()(x)
    x = Dense(2, kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x, name='side_out')
    #
    main = Dense(10, activation='softmax', name='main_out', kernel_regularizer=l2(weight_decay))(x)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, labels])
    return main, side


### run model


def run(lambda_centerloss):
    """
    Run the model
    :param lambda_centerloss:
    :return:
    """

    ### get data

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)

    ### compile

    main_input = Input((28, 28, 1))
    aux_input = Input((10,))

    final_output, side_output = my_model(main_input, aux_input)

    model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])
    model.summary()

    optim = optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, lambda_centerloss])

    ### callbacks

    utils.build_empty_dir('logs')
    # utils.build_empty_dir('images-lambda-{}'.format(lambda_centerloss))
    call1 = TensorBoard(log_dir='logs')
    # call2 = my_callbacks.CenterLossCall(lambda_centerloss)
    call3 = my_callbacks.Alpha_Print()

    ### fit

    dummy1 = np.zeros((x_train.shape[0], 1))
    dummy2 = np.zeros((x_test.shape[0], 1))

    model.fit([x_train, y_train_onehot], [y_train_onehot, dummy1], batch_size=batch_size,
              epochs=epochs,
              verbose=1, validation_data=([x_test, y_test_onehot], [y_test_onehot, dummy2]),
              callbacks=[call1, call3])

    ### run training and val sets

    reduced_model = Model(inputs=model.input[0], outputs=model.get_layer('side_out').output)
    feats = reduced_model.predict(x_train)
    my_callbacks.visualize_train(feats, y_train, epoch=epochs - 1,
                                 centers=model.get_layer('centerlosslayer').get_weights()[0],
                                 lambda_cl=lambda_centerloss)
    feats = reduced_model.predict(x_test)
    my_callbacks.visualize(feats, y_test, epoch=epochs - 1,
                                 centers=model.get_layer('centerlosslayer').get_weights()[0],
                                 lambda_cl=lambda_centerloss)

    ### done

    K.clear_session()
    return


###

if __name__ == '__main__':
    run(0.1)
