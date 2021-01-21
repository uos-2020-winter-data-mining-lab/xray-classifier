"""
Retrain the YOLO model for your own dataset.
"""
import os
from keras.models import Model
from keras.layers import UpSampling2D, Concatenate
from keras.layers import Input, Lambda, Conv2D, ZeroPadding2D
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from .yolov3 import yolo_loss


def create_model(
    image_input, anchors, num_classes, weights="data/yolov3-320.h5",
    summary=True
):
    '''create the training model'''
    infer_model = make_yolov3_model(
        image_input=image_input,
        anchors=anchors,
        num_classes=num_classes,
        max_box_per_image=30
    )

    if os.path.exists(weights):
        print(f'>> Load weights "{weights}"')
        infer_model.load_weights(weights, by_name=True, skip_mismatch=True)
        layers = len(infer_model.layers)
        frozen_num = layers - 3
        for i in range(frozen_num):
            infer_model.layers[i].trainable = False

    h, w = image_input
    num_anchors = len(anchors)
    y_true = [
        Input(shape=(
            h//{0: 32, 1: 16, 2: 8}[layer],
            w//{0: 32, 1: 16, 2: 8}[layer],
            num_anchors // 3,
            num_classes + 5)
        ) for layer in range(3)
    ]

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'ignore_thresh': 0.5
        }
    )([*infer_model.output, *y_true])

    train_model = Model([infer_model.input, *y_true], model_loss)

    train_model.compile(
        optimizer=Adam(lr=1e-4),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    if summary:
        train_model.summary(positions=[.35, .65, .73, 1.])

    return [train_model, infer_model]


def make_yolov3_model(image_input, anchors, num_classes, max_box_per_image):
    image_input = Input(shape=(None, None, 3))

    # INPUT BLOCKS
    x = YOLO_Conv(image_input, [
            YOLOLayer(32, kernel=3, stride=1, BN=True, LRU=True, id=1),
            YOLOLayer(64, kernel=3, stride=2, BN=True, LRU=True, id=2),
            YOLOLayer(32, kernel=1, stride=1, BN=True, LRU=True, id=3),
            YOLOLayer(64, kernel=3, stride=1, BN=True, LRU=True, id=4)])

    # RESIDUAL BLOCKS
    x = make_residual_blocks(x, filters=128,  num_blocks=2, id=5)
    x = x36 = make_residual_blocks(x, filters=256,  num_blocks=8, id=12)
    x = x61 = make_residual_blocks(x, filters=512,  num_blocks=8, id=37)
    x = make_residual_blocks(x, filters=1024, num_blocks=4, id=62)

    # OUTPUT BLOCKS
    classes = 3 * (5 + num_classes)
    x, output1 = make_last_layers(x, 512, classes, skip_block=None, id=75)
    x, output2 = make_last_layers(x, 256, classes, skip_block=x61,  id=84)
    x, output3 = make_last_layers(x, 128, classes, skip_block=x36,  id=96)

    infer_model = Model(image_input, [output1, output2, output3])

    return infer_model


def make_last_layers(x, filters, classes, skip_block=None, id=None):
    if skip_block:
        x = YOLO_Conv(x, [
            YOLOLayer(classes, kernel=1, stride=1, BN=True, LRU=True, id=id)],
            skip=False)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, skip_block])
        id = id + 3

    x = YOLO_Conv(x, [
        YOLOLayer(filters,   kernel=1, stride=1, BN=True, LRU=True, id=id),
        YOLOLayer(filters*2, kernel=3, stride=1, BN=True, LRU=True, id=id+1),
        YOLOLayer(filters,   kernel=1, stride=1, BN=True, LRU=True, id=id+2),
        YOLOLayer(filters*2, kernel=3, stride=1, BN=True, LRU=True, id=id+3),
        YOLOLayer(filters,   kernel=1, stride=1, BN=True, LRU=True, id=id+4)],
        skip=False)

    outputs = YOLO_Conv(x, [
        YOLOLayer(filters*2,  kernel=3, stride=1, BN=True, LRU=True, id=id+5),
        YOLOLayer(classes, kernel=1, stride=1, id=id+6)],
        skip=False)

    return x, outputs


def make_residual_blocks(inputs, filters, num_blocks, id):
    inputs = YOLO_Conv(inputs, [
        YOLOLayer(filters,    kernel=3, stride=2, BN=True, LRU=True, id=id),
        YOLOLayer(filters//2, kernel=1, stride=1, BN=True, LRU=True, id=id+1),
        YOLOLayer(filters,    kernel=3, stride=1, BN=True, LRU=True, id=id+2)])

    id = id + 3
    for i in range(num_blocks-1):
        inputs = YOLO_Conv(inputs, [
            YOLOLayer(filters//2, kernel=1, stride=1, BN=True, LRU=True, id=id+i*3),
            YOLOLayer(filters,    kernel=3, stride=1, BN=True, LRU=True, id=id+i*3+1)])

    return inputs


def YOLOLayer(filter, kernel, stride, BN=False, LRU=False, id=None):
    layer_info = {
        'filter': filter,
        'kernel': kernel,
        'stride': stride,
        'BN': BN,
        'LRU': LRU,
        'idx': id
    }
    return layer_info


def YOLO_Conv(inputs, layers, skip=True):
    x = inputs
    count = 0

    for layer in layers:
        if count == (len(layers) - 2) and skip:
            skip_connection = x
        count += 1

        if layer['stride'] > 1:
            # peculiar padding as darknet prefer left and top
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(layer['filter'],
                   layer['kernel'],
                   strides=layer['stride'],
                   # peculiar padding as darknet prefer left and top
                   padding='valid' if layer['stride'] > 1 else 'same',
                   name='conv_' + str(layer['idx']),
                   use_bias=False if layer['BN'] else True)(x)
        if layer['BN']:
            x = BatchNormalization(
                epsilon=0.001, name='BN_' + str(layer['idx']))(x)
        if layer['LRU']:
            x = LeakyReLU(alpha=0.1, name='LRU_' + str(layer['idx']))(x)

    return add([skip_connection, x]) if skip else x
