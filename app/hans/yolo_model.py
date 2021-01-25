"""
Retrain the YOLO model for your own dataset.
"""
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from .yolo_block import (
    make_last_layers, make_residual_blocks, yolo_block, yolo_layer
)
from .yolo_layer import YoloLayer


def create_model(
    num_classes,
    anchors,
    max_box_per_image,
    max_grid,
    batch_size,
    warmup_batches,
    ignore_thresh,
    multi_gpu,
    weights,
    learning_rate,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale
):
    '''create the training model'''
    train_model, infer_model = make_yolov3_model(
        num_classes=num_classes,
        anchors=anchors,
        max_box_per_image=max_box_per_image,
        max_grid=max_grid,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        ignore_thresh=ignore_thresh,
        grid_scales=grid_scales,
        obj_scale=obj_scale,
        noobj_scale=noobj_scale,
        xywh_scale=xywh_scale,
        class_scale=class_scale
    )

    if os.path.exists(weights):
        print(f"pretrained weights {weights} is loaded")
        train_model.load_weights(weights, by_name=True)

    train_model.compile(loss=dummy_loss, optimizer=Adam(lr=learning_rate))

    return [train_model, infer_model]


def make_yolov3_model(
    num_classes,
    anchors,
    max_box_per_image,
    max_grid,
    batch_size,
    warmup_batches,
    ignore_thresh,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale
):
    image_input = Input(shape=(None, None, 3))
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo1 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes))
    true_yolo2 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes))
    true_yolo3 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes))

    # INPUT BLOCKS
    x = yolo_block(image_input, [
        yolo_layer(32, kernel=3, stride=1, BN=True, LRU=True, id=1),
        yolo_layer(64, kernel=3, stride=2, BN=True, LRU=True, id=2),
        yolo_layer(32, kernel=1, stride=1, BN=True, LRU=True, id=3),
        yolo_layer(64, kernel=3, stride=1, BN=True, LRU=True, id=4)])

    # RESIDUAL BLOCKS
    x = make_residual_blocks(x, filters=128,  num_blocks=2, id=5)
    x = x_36 = make_residual_blocks(x, filters=256, num_blocks=8, id=12)
    x = x_61 = make_residual_blocks(x, filters=512, num_blocks=8, id=37)
    x = make_residual_blocks(x, filters=1024, num_blocks=4, id=62)

    # OUTPUT BLOCKS
    classes = 3 * (5 + num_classes)
    x, output1 = make_last_layers(x, 512, classes, skip_block=None, id=75)
    loss1 = YoloLayer(
        anchors[12:],
        [1*num for num in max_grid],
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales[0],
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
    )([x, output1, true_yolo1, true_boxes])

    x, output2 = make_last_layers(x, 256, classes, skip_block=x_61, id=84)
    loss2 = YoloLayer(
        anchors[6:12],
        [2*num for num in max_grid],
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales[1],
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
    )([x, output2, true_yolo2, true_boxes])

    x, output3 = make_last_layers(x, 128, classes, skip_block=x_36, id=96)
    loss3 = YoloLayer(
        anchors[:6],
        [4*num for num in max_grid],
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales[2],
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
    )([x, output3, true_yolo3, true_boxes])

    infer_model = Model(image_input, [output1, output2, output3])
    train_model = Model(
        [image_input, true_boxes, true_yolo1, true_yolo2, true_yolo3],
        [loss1, loss2, loss3])

    return train_model, infer_model


def dummy_loss(y_true, y_pred):
    loss = tf.sqrt(tf.reduce_sum(y_pred))
    return loss
