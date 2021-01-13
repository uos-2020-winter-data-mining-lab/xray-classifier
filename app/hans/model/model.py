import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from app.hans.config import WIDTH, HEIGHT, RATIO


def set_model(summary=False):
    model = BasicModel
    return model


class BasicModel():
    def __init__(self):
        INPUT_SHAPE = (WIDTH//RATIO, HEIGHT//RATIO, 3)
        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE)
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5, activation='softmax'))

        model.summary()

        return model


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    pad_shape = []
    if data_format == 'channels_first':
        pad_shape = [
            [0, 0],
            [0, 0],
            [pad_beg, pad_end],
            [pad_beg, pad_end]
        ]
    else:
        pad_shape = [
            [0, 0],
            [pad_beg, pad_end],
            [pad_beg, pad_end],
            [0, 0]
        ]

    padded_inputs = tf.pad(inputs, pad_shape)
    return padded_inputs


def Conv2D_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return Conv2D(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format
    )
