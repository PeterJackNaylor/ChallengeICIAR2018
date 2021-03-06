# -*- coding: utf-8 -*-

import pdb
from optparse import OptionParser
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss, accuracy_score
import sys
from sklearn.model_selection import StratifiedKFold
import numpy as np
# from load_cifar10 import load_cifar10_data
from load_dataICIAR import load_ICIAR_data
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append('..')
from utils.GeneratorKeras import ICIARSequence, ICIARSequenceTest
from keras.callbacks import EarlyStopping, ModelCheckpoint


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None, lr=1e-3, mom=0.9, w_d=1e-6):
    """
    Resnet 50 Model for Keras

    Model Schema is based on 
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data 
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/resnet50_weights_th_dim_ordering_th_kernels.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=lr, decay=w_d, momentum=mom, nesterov=True)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=w_d, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  
    return model

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10
    parser = OptionParser()
    parser.add_option('--lr', dest="lr", type="float")
    parser.add_option('--mom', dest="momentum", type="float")
    parser.add_option('--weight_decay', dest="weight_decay", type="float")
    parser.add_option('--output', dest="output", type="str")
    parser.add_option('--output_mod', dest="output_mod", type="str")
    parser.add_option('--split', dest="split", type="int")
    parser.add_option('--epoch', dest="epoch", type="int")
    parser.add_option('--bs', dest="bs", type="int")
    parser.add_option('--mean', dest="mean", type="str")
    parser.add_option('--k', dest="k", type="int")
    parser.add_option('--path', dest="path", type="str")
    (options, args) = parser.parse_args()

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 4
    batch_size = options.bs
    nb_epoch = options.epoch

    k = options.k
    train_datagen = ICIARSequence(options.path, k, num_classes, batch_size)
    valid_datagen = ICIARSequenceTest(options.path, k, num_classes, batch_size)
    # Load our model
    lr = options.lr
    mom = options.momentum
    w_d = options.weight_decay
    model = resnet50_model(img_rows, img_cols, channel, num_classes, lr)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, \
                               verbose=1, mode='auto')
    fold_name = options.output_mod
    fold_name = fold_name.replace('.h5', '_fold_{}.h5').format(k)
    best_model_saver = ModelCheckpoint(fold_name, 
                                        monitor='val_loss', 
                                        verbose=0,
                                        save_best_only=True, 
                                        save_weights_only=False,
                                        mode='auto',
                                        period=1)
    callbacks_list = [earlystop, best_model_saver]


    model.fit_generator(train_datagen, 
                        steps_per_epoch=train_datagen.__len__(), 
                        class_weight=train_datagen.weight_dic(),
                        epochs=nb_epoch,
                        max_queue_size=20, 
                        workers=10, 
                        use_multiprocessing=False,
                        validation_data=valid_datagen,
                        validation_steps=valid_datagen.__len__(),
                        verbose=1,
                        callbacks=callbacks_list)

    loss, acc = model.evaluate_generator(valid_datagen, max_queue_size=1000, workers=16)
    loss = np.array([loss])
    acc = np.array([acc])

    pd.DataFrame({'cross-entropy': loss, 'accuracy': acc}).to_csv(options.output)

