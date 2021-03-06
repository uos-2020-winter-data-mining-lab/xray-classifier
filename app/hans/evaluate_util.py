import os
import cv2
import numpy as np
from scipy.special import expit
from app.hans.bbox import BoundBox
from app.hans.colors import get_color


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape
    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[
        (net_h-new_h)//2:(net_h+new_h)//2,
        (net_w-new_w)//2:(net_w+new_w)//2,
        :
    ] = resized
    new_image = np.expand_dims(new_image, 0)
    return new_image


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row][col][b][4]
            if(objectness <= obj_thresh):
                continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row][col][b][:4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)

    return boxes


def _sigmoid(x):
    return expit(x)


def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        num_class = len(boxes[0].classes)
    else:
        return

    for c in range(num_class):
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


def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '':
                    label_str += ', '

                box_score = round(box.get_score()*100, 2)
                label_str += f"{labels[i]} {str(box_score)}%"
                label = i

            if not quiet:
                print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(
                label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5
            )
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([
                [box.xmin-1, box.ymin],
                [box.xmin-1, box.ymin-height-10],
                [box.xmin+width, box.ymin-height-10],
                [box.xmin+width, box.ymin]
            ], dtype='int32')

            cv2.rectangle(
                image,
                pt1=(box.xmin, box.ymin),
                pt2=(box.xmax, box.ymax),
                color=get_color(label),
                thickness=2
            )
            cv2.fillPoly(image, pts=[region], color=get_color(label))
            cv2.putText(
                image,
                text=label_str,
                org=(box.xmin+5, box.ymin-7),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1e-3 * image.shape[0],
                color=(0, 0, 0),
                thickness=1
            )

    num = len(os.listdir('imgs'))
    file_name = f'imgs/detected_{num}.png'
    cv2.imwrite(file_name, image)
    return image
