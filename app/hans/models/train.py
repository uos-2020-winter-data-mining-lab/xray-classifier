"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from .model import yolo_body, yolo_loss
from app.hans.config import INPUT_SHAPE, NUM_CLASSES


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


def create_model(
    anchors, load_pretrained=True, freeze_body=2, weights='data/yolov3.h5'
):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = INPUT_SHAPE
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(
            h//{0: 32, 1: 16, 2: 8}[layer],
            w//{0: 32, 1: 16, 2: 8}[layer],
            num_anchors // 3,
            NUM_CLASSES + 5)
        ) for layer in range(3)
    ]

    model_body = yolo_body(image_input, num_anchors//3, NUM_CLASSES)
    print(f'Create YOLOv3 model with {num_anchors} anchors '
          f'and {NUM_CLASSES} classes.')

    if load_pretrained is True:
        model_body.load_weights(weights, by_name=True, skip_mismatch=True)
        print(f'Load weights "{weights}"')
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print(f'Freeze the First {num} layers of '
                  f'total {len(model_body.layers)}')

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={
            'anchors': anchors,
            'num_classes': NUM_CLASSES,
            'ignore_thresh': 0.5
        }
    )([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    return model
