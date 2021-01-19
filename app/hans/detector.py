import os
import colorsys
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.layers import Input
from timeit import default_timer as timer
from app.hans.models.utils import letterbox_image
from app.hans.models.yolov3 import yolo_eval, yolo_body
from app.hans.process.process import crop_and_resize_images
from app.hans.models.model import create_model
from PIL import Image, ImageFont, ImageDraw


def detect_img():
    yolo = YOLO()
    while True:
        img = input('Input image filename:')
        try:
            image = yolo.detect_image(img)
            image.show()
        except Exception as e:
            print(f'{e} \n Open Error! Try again!')
            continue
    yolo.close_session()


class YOLO(object):
    _defaults = {
        "model_path": 'data/trained_weights_final.h5',
        "anchors_path": 'data/metadata/anchors.txt',
        "classes_path": 'data/metadata/classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.compat.v1.Session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5')

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except Exception:
            print('exception')
            self.yolo_model = yolo_body(
                Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print(f'{model_path} model,'
              f'{num_anchors} anchors,'
              f'and {num_classes} classes loaded.')

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        # Fixed seed for consistent colors across runs.
        np.random.seed(10101)
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(self.colors)
        # Reset seed to default.
        np.random.seed(None)

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
            yolo_outputs=self.yolo_model.output,
            anchors=self.anchors,
            num_classes=len(self.class_names),
            image_shape=self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou
        )
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        image = crop_and_resize_images(image)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, \
                   'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, \
                   'Multiples of 32 required'
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        model = create_model(
            input_shape=(416, 416),
            anchors=self.anchors,
            num_classes=len(self.class_names),
            freeze_body=2,
            weights='data/yolov3-320.h5',
            summary=False
        )

        yolo_outputs = model.predict(image_data)

        # out_boxes, out_scores, out_classes = yolo_eval(
        #     yolo_outputs
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image.size[1], image.size[0]],
        #         K.learning_phase(): 0
        #     }
        # )

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # font = ImageFont.truetype(
        #     font='font/FiraMono-Medium.otf',
        #     size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 300

        # for i, c in reversed(list(enumerate(out_classes))):
        #     predicted_class = self.class_names[c]
        #     box = out_boxes[i]
        #     score = out_scores[i]

        #     label = '{} {:.2f}'.format(predicted_class, score)
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)

        #     top, left, bottom, right = box
        #     top = max(0, np.floor(top + 0.5).astype('int32'))
        #     left = max(0, np.floor(left + 0.5).astype('int32'))
        #     bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        #     right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #     print(label, (left, top), (right, bottom))

        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])

        #     # My kingdom for a good redistributable image drawing library.
        #     for i in range(thickness):
        #         draw.rectangle(
        #             [left + i, top + i, right - i, bottom - i],
        #             outline=self.colors[c])
        #     draw.rectangle(
        #         [tuple(text_origin), tuple(text_origin + label_size)],
        #         fill=self.colors[c])
        #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #     del draw

        # end = timer()
        # print(end - start)
        # return image

    def close_session(self):
        self.sess.close()
