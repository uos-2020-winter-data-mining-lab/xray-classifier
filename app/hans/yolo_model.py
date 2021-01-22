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
    lr,
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
        print("pretrained weights is loaded")
        # train_model.load_weights(weights, by_name=True)

    train_model.compile(loss=dummy_loss, optimizer=Adam(lr=1e-4))

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
    loss1 = YOLO_loss(
        [x, output1, true_yolo1, true_boxes], anchors[12:],
        [1 * num for num in max_grid], grid_scales[0])

    x, output2 = make_last_layers(x, 256, classes, skip_block=x_61, id=84)
    loss2 = YOLO_loss(
        [x, output2, true_yolo2, true_boxes], anchors[6:12],
        [2 * num for num in max_grid], grid_scales[1])

    x, output3 = make_last_layers(x, 128, classes, skip_block=x_36, id=96)
    loss3 = YOLO_loss(
        [x, output3, true_yolo3, true_boxes], anchors[:6],
        [4 * num for num in max_grid], grid_scales[2])

    infer_model = Model(image_input, [output1, output2, output3])
    train_model = Model(
        [image_input, true_boxes, true_yolo1, true_yolo2, true_yolo3],
        [loss1, loss2, loss3])

    return train_model, infer_model


def YOLO_loss(
    x,
    anchors,
    max_grid,
    grid_scale,
    batch_size=16,
    warmup_batches=3,
    ignore_thresh=0.5,
    obj_scale=5,
    noobj_scale=1,
    xywh_scale=1,
    class_scale=1,
):
    input_image, y_pred, y_true, true_boxes = x

    # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 5+nb_class]
    y_pred = tf.reshape(y_pred, tf.concat(
        [tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

    # initialize the masks
    object_mask = tf.expand_dims(y_true[..., 4], 4)

    # the variable to keep track of number of batches processed
    batch_seen = tf.Variable(0.)

    # compute grid factor and net factor
    grid_h = tf.shape(y_true)[1]
    grid_w = tf.shape(y_true)[2]
    grid_factor = tf.reshape(
        tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

    net_h = tf.shape(input_image)[1]
    net_w = tf.shape(input_image)[2]
    net_factor = tf.reshape(
        tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])
    """
    Adjust prediction
    """
    max_grid_h, max_grid_w = max_grid
    cell_x = to_float(tf.reshape(tf.tile(
        tf.range(max_grid_w), [max_grid_h]),
        (1, max_grid_h, max_grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    cell_grid = tf.tile(
        tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])
    pred_box_xy = (
        cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))
    # t_wh
    pred_box_wh = y_pred[..., 2:4]
    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)
    # adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
    true_box_wh = y_true[..., 2:4]  # t_wh
    true_box_conf = tf.expand_dims(y_true[..., 4], 4)
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Compare each predicted box to all true boxes
    """
    # initially, drag all objectness of all boxes to 0
    conf_delta = pred_box_conf - 0

    # then, ignore the boxes which have good overlap with some true box
    true_xy = true_boxes[..., 0:2] / grid_factor
    true_wh = true_boxes[..., 2:4] / net_factor

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
    pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
    pred_wh = tf.expand_dims(
        tf.exp(pred_box_wh) * anchors / net_factor, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_delta *= tf.expand_dims(to_float(best_ious < ignore_thresh), 4)

    """
    Compute some online statistics
    """
    true_xy = true_box_xy / grid_factor
    true_wh = tf.exp(true_box_wh) * anchors / net_factor

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = pred_box_xy / grid_factor
    pred_wh = tf.exp(pred_box_wh) * anchors / net_factor

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

    """
    Warm-up training
    """
    batch_seen = tf.compat.v1.assign_add(batch_seen, 1.)

    true_box_xy, true_box_wh, xywh_mask = tf.cond(
        tf.less(batch_seen, warmup_batches+1),
        lambda: [
            true_box_xy + (0.5 + cell_grid[:, :grid_h, :grid_w, :, :]) * (1-object_mask),
            true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), tf.ones_like(object_mask)
        ],
        lambda: [true_box_xy, true_box_wh, object_mask])

    """
    Compare each true box to all anchor boxes
    """
    wh_scale = tf.exp(true_box_wh) * anchors / net_factor
    # the smaller the box, the bigger the scale
    wh_scale = tf.expand_dims(
        2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)

    xy_delta = xywh_mask * (pred_box_xy-true_box_xy) * \
        wh_scale * xywh_scale
    wh_delta = xywh_mask * (pred_box_wh-true_box_wh) * \
        wh_scale * xywh_scale
    conf_delta = object_mask * (pred_box_conf-true_box_conf) * \
        obj_scale + (1-object_mask) * conf_delta * noobj_scale
    class_delta = object_mask * \
        tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
        class_scale

    loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
    loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
    loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
    loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))

    loss = loss_xy + loss_wh + loss_conf + loss_class

    ret = loss * grid_scale
    return ret


def dummy_loss(y_true, y_pred):
    loss = tf.sqrt(tf.reduce_sum(y_pred))
    return loss


def to_float(x):
    return tf.cast(x, dtype=tf.float32)
