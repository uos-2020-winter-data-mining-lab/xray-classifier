import os
import numpy as np
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from app.hans.config import INPUT_SHAPE
from .models.train import create_model

DATA_DIR = 'data/'
RAW_DATA = f'{DATA_DIR}/raw/dataset'


def run():
    # Step 1 . Path
    annotation_path = os.path.join(DATA_DIR, 'annotation.txt')
    with open(annotation_path) as f:
        lines = f.readlines()

    classes_path = os.path.join(DATA_DIR, 'labels.txt')
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    anchors_path = os.path.join(DATA_DIR, 'anchors.txt')
    anchors = get_anchors(anchors_path)

    # Step 1 . Model Config
    input_shape = INPUT_SHAPE

    # reduce_lr =
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # early_stopping =
    EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    model = create_model(input_shape, anchors, num_classes, freeze_body=2)
    model.summary()

    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    print(num_val)
    print(num_train)

    # # Step0. config
    # RAW_DIR = os.path.join(DATA_DIR, 'raw\\dataset\\')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'train\\')
    # VALID_DIR = os.path.join(DATA_DIR, 'valid\\')
    # TEST_DIR = os.path.join(DATA_DIR, 'test\\')

    # # Step1. Data Preparing
    # init_flag = False
    # print("DATA COPYING...")
    # if init_flag is True:
    #     copy_data(RAW_DIR)

    #     preprocess_data(TRAIN_DIR)
    #     preprocess_data(VALID_DIR)

    # # Step2. Data Generating
    # print("DATA GENERATING...")
    # train_generator = generator(TRAIN_DIR)
    # valid_generator = generator(VALID_DIR)

    # print("MODEL SETTING...")
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(192, 108, 3)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(5, activation='softmax'))
    # model.summary()
    # # Step3. Build a Model
    # for i in range(len(model.layers)):
    #     model.layers[i].trainable = True

    # print("MODEL COMPILE...")
    # # use custom yolo_loss Lambda layer.
    # # {'yolo_loss': lambda y_true, y_pred: y_pred},
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=1e-4),
    #               metrics=['acc'])

    # # Step4. Fit the Model
    # print("MODEL FITTING...")
    # print(f'Train on {num_train} samples,'
    #       f' val on {num_val} samples,'
    #       f' with batch size {batch_size}.')

    # history = model.fit(
    #     data_generator_wrapper(
    #         lines[:num_train], batch_size, input_shape, anchors, num_classes),
    #     steps_per_epoch=max(1, num_train//batch_size),
    #     validation_data=data_generator_wrapper(
    #         lines[num_train:], batch_size, input_shape, anchors, num_classes),
    #     validation_steps=max(1, num_val//batch_size),
    #     epochs=100,
    #     callbacks=[reduce_lr, early_stopping]
    # )

    # # Step5. Predict / Evaluate
    # # results = model.evaluate(test_data, test_labels)

    # # Step6. Plotting
    # plotting(history)

    # # Step7. Output
    # print("RUN XRAY DATA \n epochs : {EPOCHS} \n")


if __name__ == '__main__':
    run()


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator_wrapper(
    annotation_lines, batch_size, input_shape, anchors, num_classes
):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(
        annotation_lines, batch_size, input_shape, anchors, num_classes)


def data_generator(
    annotation_lines, batch_size, input_shape, anchors, num_classes
):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(
                annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(
            box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4] < num_classes).all(), \
        'class id must be less than num_classes'

    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
    ] if num_layers == 3 else [
        [3, 4, 5],
        [1, 2, 3]
    ]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [
        input_shape // {
            0: 32,
            1: 16,
            2: 8
        }[layer] for layer in range(num_layers)
    ]
    y_true = [
        np.zeros(
            (m,
             grid_shapes[layer][0],
             grid_shapes[layer][1],
             len(anchor_mask[layer]),
             5+num_classes),
            dtype='float32'
        ) for layer in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for layer in range(num_layers):
                if n in anchor_mask[layer]:
                    i = np.floor(
                        true_boxes[b, t, 0] * grid_shapes[layer][1]
                    ).astype('int32')
                    j = np.floor(
                        true_boxes[b, t, 1] * grid_shapes[layer][0]
                    ).astype('int32')
                    k = anchor_mask[layer].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[layer][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[layer][b, j, i, k, 4] = 1
                    y_true[layer][b, j, i, k, 5+c] = 1

    return y_true


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data(
    annotation_line, input_shape, random=True, max_boxes=20,
    jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True
):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array(
        [np.array(list(map(int, box.split(',')))) for box in line[1:]]
    )

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1-jitter, 1+jitter) / rand(1-jitter, 1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
