"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
from keras.models import Model
from keras.layers import UpSampling2D, Concatenate
from keras.layers import Input, Conv2D, ZeroPadding2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def create_model(
    image_input, anchors, num_classes, max_grid, weights="data/yolov3-320.h5",
    summary=True
):
    '''create the training model'''
    train_model, infer_model = make_yolov3_model(
        image_input=image_input,
        anchors=anchors,
        num_classes=num_classes,
        max_grid=max_grid,
        grid_scales=[1, 1, 1],
        max_box_per_image=30
    )
    train_model.load_weights(weights, by_name=True)
    train_model.compile(loss=dummy_loss, optimizer=Adam(lr=1e-4))

    return [train_model, infer_model]


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))


def make_yolov3_model(
    image_input, anchors, num_classes, max_box_per_image, max_grid,
    grid_scales
):
    image_input = Input(shape=(None, None, 3))
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo1 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes))
    true_yolo2 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes))
    true_yolo3 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes))

    # INPUT BLOCKS
    x = YOLO_Conv(image_input, [
            YOLOLayer(32, kernel=3, stride=1, BN=True, LRU=True, id=1),
            YOLOLayer(64, kernel=3, stride=2, BN=True, LRU=True, id=2),
            YOLOLayer(32, kernel=1, stride=1, BN=True, LRU=True, id=3),
            YOLOLayer(64, kernel=3, stride=1, BN=True, LRU=True, id=4)])

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
        [1*num for num in max_grid], grid_scales[0])

    x, output2 = make_last_layers(x, 256, classes, skip_block=x_61, id=84)
    loss2 = YOLO_loss(
        [x, output2, true_yolo2, true_boxes], anchors[6:12],
        [2*num for num in max_grid], grid_scales[1])

    x, output3 = make_last_layers(x, 128, classes, skip_block=x_36, id=96)
    loss3 = YOLO_loss(
        [x, output3, true_yolo3, true_boxes], anchors[:6],
        [4*num for num in max_grid], grid_scales[2])

    infer_model = Model(image_input, [output1, output2, output3])
    train_model = Model(
        [image_input, true_boxes, true_yolo1, true_yolo2, true_yolo3],
        [loss1, loss2, loss3])

    return train_model, infer_model


def make_last_layers(x, filters, classes, skip_block=None, id=None):
    if skip_block is not None:
        x = YOLO_Conv(x, [
            YOLOLayer(filters, kernel=1, stride=1, BN=True, LRU=True, id=id)],
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
        if skip and count == (len(layers) - 2):
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
    # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
    y_pred = tf.reshape(y_pred, tf.concat(
        [tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

    # initialize the masks
    object_mask = tf.expand_dims(y_true[..., 4], 4)

    # the variable to keep track of number of batches processed
    batch_seen = tf.Variable(0.)

    anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])

    # compute grid factor and net factor
    grid_h = tf.shape(y_true)[1]
    grid_w = tf.shape(y_true)[2]
    grid_factor = tf.reshape(to_float([grid_w, grid_h]), [1, 1, 1, 1, 2])

    net_h = tf.shape(input_image)[1]
    net_w = tf.shape(input_image)[2]
    net_factor = tf.reshape(to_float([net_w, net_h]), [1, 1, 1, 1, 2])

    """
    Adjust prediction
    """
    max_grid_h, max_grid_w = max_grid
    cell_x = to_float(
        tf.reshape(
            tf.tile(tf.range(max_grid_w), [max_grid_h]),
            (1, max_grid_h, max_grid_w, 1, 1)
        ))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(
        tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

    # sigma(t_xy) + c_xy
    pred_box_xy = \
        (cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))
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

    count = tf.reduce_sum(object_mask)
    count_noobj = tf.reduce_sum(1 - object_mask)
    detect_mask = to_float((pred_box_conf*object_mask) >= 0.5)
    class_mask = tf.expand_dims(
        to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
    recall50 = tf.reduce_sum(
        to_float(iou_scores >= 0.5) * detect_mask * class_mask
    ) / (count + 1e-3)
    recall75 = tf.reduce_sum(
        to_float(iou_scores >= 0.75) * detect_mask * class_mask
    ) / (count + 1e-3)
    avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
    avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
    avg_noobj = tf.reduce_sum(
        pred_box_conf * (1-object_mask)) / (count_noobj + 1e-3)
    avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

    """
    Warm-up training
    """
    batch_seen = tf.compat.v1.assign_add(batch_seen, 1.)

    true_box_xy, true_box_wh, xywh_mask = tf.cond(
        tf.less(batch_seen, warmup_batches+1),
        lambda: [
            true_box_xy + (0.5 + cell_grid[:, :grid_h, :grid_w, :, :]) * (1-object_mask),
            true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), tf.ones_like(object_mask)],
        lambda: [true_box_xy, true_box_wh, object_mask]
    )

    """
    Compare each true box to all anchor boxes
    """
    wh_scale = tf.exp(true_box_wh) * anchors / net_factor
    # the smaller the box, the bigger the scale
    wh_scale = tf.expand_dims(
        2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)
    xy_delta = xywh_mask * (pred_box_xy-true_box_xy) * wh_scale * xywh_scale
    wh_delta = xywh_mask * (pred_box_wh-true_box_wh) * wh_scale * xywh_scale
    conf_delta = object_mask * (pred_box_conf-true_box_conf) * \
        obj_scale + (1-object_mask) * conf_delta * noobj_scale
    class_delta = object_mask * \
        tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
        class_scale
    loss_xy = tf.reduce_sum(tf.square(xy_delta),       list(range(1, 5)))
    loss_wh = tf.reduce_sum(tf.square(wh_delta),       list(range(1, 5)))
    loss_conf = tf.reduce_sum(tf.square(conf_delta),     list(range(1, 5)))
    loss_class = tf.reduce_sum(
        class_delta,               list(range(1, 5)))
    loss = loss_xy + loss_wh + loss_conf + loss_class
    """
    if debug:
        loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
                               tf.reduce_sum(loss_wh),
                               tf.reduce_sum(loss_conf),
                               tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)
    """
    return loss * grid_scale


def to_float(x):
    return tf.cast(x, dtype=tf.float32)
