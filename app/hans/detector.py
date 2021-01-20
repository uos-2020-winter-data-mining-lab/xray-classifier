import os
import colorsys
import numpy as np
import cv2
import traceback
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
    path = "data/img/dataset/Astrophysics/Aerosol/Single_Default/"
    img = path + "H_8481.80-1090_01_153.png"
    try:
        image = yolo.detect_image(img)
        image.show()
    except Exception as e:
        print(f'{e} \n Open Error! Try again!')


class YOLO(object):
    _defaults = {
        "model_path": 'data/yolo_trained_weights.h5',
        "anchors_path": 'data/metadata/anchors.txt',
        "classes_path": 'data/metadata/classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (320, 320),
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
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
            print("Calling Yolo Model")
            self.yolo_model = load_model(model_path, compile=False)
        except Exception:
            print("Calling Yolo Model Weights")
            self.yolo_model = yolo_body(
                Input(shape=(None, None, 3)), num_anchors//3, num_classes)
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

        yolos = self.yolo_model.predict(image_data)
        boxes = []

        print(len(boxes))
        for i in range(len(yolos)):
            print(">>", i)
            boxes += decode_netout(
                yolos[i][0],
                anchors=self.anchors,
                obj_thresh=self.score,
                nms_thresh=self.iou,
                image_shape=self.input_image_shape
            )
            print(len(boxes))

        do_nms(boxes, self.iou)
        print(len(boxes))
        image = draw_boxes(image, boxes, self.class_names, self.iou)
        print(len(boxes))

        image.save('detected.png')
        print("done")

        """
        print(self.yolo_model.output)
        print(self.anchors)
        print(self.class_names)
        print(self.input_image_shape)
        print(self.score)
        print(self.iou)
        out_boxes, out_scores, out_classes = yolo_eval(
            yolo_outputs=self.yolo_model.output,
            anchors=self.anchors,
            num_classes=len(self.class_names),
            image_shape=self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou
        )

        # print(f'Found {out_boxes.shape[1]} boxes for {image}')

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = f'{predicted_class}{score}'
            print(label)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        # end = timer()
        # print(end - start)
        """
        return image


def decode_netout(netout, anchors, obj_thresh, nms_thresh, image_shape):
    grid_h, grid_w = netout.shape[:2]
    net_h, net_w = image_shape
    nb_box = 3

    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            objectness = netout[int(row)][int(col)][b][4]

            if(objectness.all() <= obj_thresh):
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height

            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]

            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def draw_boxes(image, boxes, labels, obj_thresh):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                print(labels[i] + ': ' + str(box.classes[i]*100) + '%')

        if label >= 0:
            cv2.rectangle(
                image,
                (box.xmin, box.ymin),
                (box.xmax, box.ymax),
                (0, 255, 0),
                3
            )
            cv2.putText(
                image,
                label_str + ' ' + str(box.get_score()),
                (box.xmin, box.ymin - 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                1e-3 * image.shape[0],
                (0, 255, 0),
                2
            )
    return image


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap(
        [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap(
        [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3
