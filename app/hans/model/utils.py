import tensorflow as tf
from keras.layers import batch_normalization


_LEAKY_RELU = 0.1


def axis_by_format(data_format):
    return 1 if data_format == 'channel_first' else 3


def leaky_relu(inputs, alpha=_LEAKY_RELU):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


def batch_norm(inputs, training, data_format):
    _AXIS = 1 if data_format == 'channels_first' else 3
    _BATCH_NORM_DECAY = 0.9
    _BATCH_NORM_EPSILON = 1e-05

    return batch_normalization(
        inputs=inputs, axis=_AXIS, momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON, scale=True, training=training
    )
