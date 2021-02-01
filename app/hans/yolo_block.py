from keras.layers import UpSampling2D, Concatenate
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


def make_last_layers(x, filters, classes, skip_block=None, id=None):
    if skip_block is not None:
        x = yolo_block(x, [
            yolo_layer(filters, kernel=1, stride=1, BN=True, LRU=True, id=id)],
            skip=False)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, skip_block])
        id = id + 3

    x = yolo_block(x, [
        yolo_layer(filters,   kernel=1, stride=1, BN=True, LRU=True, id=id),
        yolo_layer(filters*2, kernel=3, stride=1, BN=True, LRU=True, id=id+1),
        yolo_layer(filters,   kernel=1, stride=1, BN=True, LRU=True, id=id+2),
        yolo_layer(filters*2, kernel=3, stride=1, BN=True, LRU=True, id=id+3),
        yolo_layer(filters,   kernel=1, stride=1, BN=True, LRU=True, id=id+4)],
        skip=False)

    outputs = yolo_block(x, [
        yolo_layer(filters*2,  kernel=3, stride=1, BN=True, LRU=True, id=id+5),
        yolo_layer(classes, kernel=1, stride=1, id=id+6)],
        skip=False)

    return x, outputs


def make_residual_blocks(inputs, filters, num_blocks, id):
    inputs = yolo_block(inputs, [
        yolo_layer(filters,    kernel=3, stride=2, BN=True, LRU=True, id=id),
        yolo_layer(filters//2, kernel=1, stride=1, BN=True, LRU=True, id=id+1),
        yolo_layer(filters,    kernel=3, stride=1, BN=True, LRU=True, id=id+2)]
    )

    id = id + 3
    for i in range(num_blocks-1):
        inputs = yolo_block(inputs, [
            yolo_layer(filters//2, kernel=1, stride=1, BN=True, LRU=True, id=id+i*3),
            yolo_layer(filters, kernel=3, stride=1, BN=True, LRU=True, id=id+i*3+1)])

    return inputs


def yolo_layer(filter, kernel, stride, BN=False, LRU=False, id=None):
    layer_info = {
        'filter': filter,
        'kernel': kernel,
        'stride': stride,
        'BN': BN,
        'LRU': LRU,
        'idx': id
    }
    return layer_info


def yolo_block(inputs, layers, skip=True):
    x = inputs

    for count, layer in enumerate(layers):
        if skip and count == (len(layers) - 2):
            skip_connection = x

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
