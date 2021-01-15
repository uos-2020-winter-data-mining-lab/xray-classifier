"""
Retrain the YOLO model for your own dataset.
"""

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda

from .model import yolo_body, yolo_loss


def create_model(
    input_shape,
    anchors,
    num_classes,
    freeze_body=2,
    weights=None,
    summary=False
):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(
            h//{0: 32, 1: 16, 2: 8}[layer],
            w//{0: 32, 1: 16, 2: 8}[layer],
            num_anchors // 3,
            num_classes + 5)
        ) for layer in range(3)
    ]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print(f'>> Create YOLOv3 model '
          f'with {num_anchors} anchors and {num_classes} classes.')

    if weights:
        print(f'>> Load weights "{weights}"')
        model_body.load_weights(weights, by_name=True, skip_mismatch=True)

        if freeze_body in [0, 1, 2]:
            # 0: Freeze None
            # 1: Freeze Darknet53 body
            # 2: Freeze All but 3 output layers
            layers = len(model_body.layers)
            freezing_num = (0, 185, layers-3)[freeze_body]
            for i in range(freezing_num):
                model_body.layers[i].trainable = False

            print(f'>> Freeze the First {freezing_num} layers of '
                  f'total {layers}')

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'ignore_thresh': 0.5
        }
    )([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    if summary is True:
        model.summary(positions=[.35, .65, .73, 1.])

    return model
