"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from .model import preprocess_true_boxes, yolo_body, yolo_loss
from .utils import get_random_data


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


def create_model(input_shape, anchors, num_classes, freeze_body=2):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(
            h//{0: 32, 1: 16, 2: 8}[layer],
            w//{0: 32, 1: 16, 2: 8}[layer],
            num_anchors//3,
            num_classes+5)
        ) for layer in range(3)
    ]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


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


def data_generator_wrapper(
    annotation_lines, batch_size, input_shape, anchors, num_classes
):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(
        annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
