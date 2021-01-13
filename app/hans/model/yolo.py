import tensorflow as tf
from .utils import (
    axis_by_format,
    leaky_relu,
    batch_norm,
)


class Yolo_v3:
    def __init__(
        self, n_classes, model_size, max_output_size,
        iou_threshold, confidence_threshold, data_format=None
    ):
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call_(self, inputs, training):

        with tf.varivale_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = inputs / 255
            axis = axis_by_format(self.data_format)

            route1, route2, route3 = Darknet53(inputs, training, self.data_format)

            route, inputs = yolo_convolution_block(inputs, filters=512, training=training, data_format=self.data_format)
            detect1 = yolo_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[6:9], img_size=self.model_size, data_format=self.data_format)
            inputs = Conv2D_fixed_padding(route, 256, kernel_size=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training, self.data_format)
            inputs = leaky_relu(inputs)

            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
            inputs = tf.concat([inputs, route2], axis=axis)

            route, inputs = yolo_convolution_block(inputs, filters=256, training=training, data_format=self.data_format)
            detect2 = yolo_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[3:6], img_size=self.model_size, data_format=self.data_format)
            inputs = Conv2D_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training, self.data_format)
            inputs = leaky_relu(inputs)

            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
            inputs = tf.concat([inputs, route1], axis=axis)

            route, inputs = yolo_convolution_block(inputs, filters=128, training=training, data_format=self.data_format)
            detect3 = yolo_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[0:3], img_size=self.model_size, data_format=self.data_format)

            inputs = tf.concat([detect1, detect2, detect3], axis=1)
            inputs = build_boxes(inputs)

            boxes_dicts = non_max_suppression(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold
            )

            return boxes_dicts
