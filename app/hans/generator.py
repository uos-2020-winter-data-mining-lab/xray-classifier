import cv2
import numpy as np
from keras.utils import Sequence
from app.hans.bbox import BoundBox
from .generator_util import (
    apply_random_scale_and_crop,
    random_distort_image,
    random_flip
)


def normalize(image):
    return image/255.


class BatchGenerator(Sequence):
    def __init__(
        self,
        data,
        anchors,
        labels,
        downsample=32,
        max_box_per_image=30,
        batch_size=16,
        min_net_size=416,
        max_net_size=416,
        net_shape=(416, 416),
        shuffle=True,
        jitter=0.1,
        norm=normalize
    ):
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.num_classes = 4 + 1 + len(self.labels)

        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.min_net_size = (min_net_size//self.downsample)*self.downsample
        self.max_net_size = (max_net_size//self.downsample)*self.downsample
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.raw_anchors = anchors
        self.anchors = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1])
                        for i in range(len(anchors)//2)]
        self.num_layers = len(self.anchors) // 3
        self.net_h, self.net_w = net_shape

        if shuffle:
            np.random.shuffle(self.data)

    def __len__(self):
        return int(np.ceil(float(len(self.data))/self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_shape = net_h, net_w = self._get_net_size(idx)
        input_shape = np.array((net_h, net_w), dtype='int32')

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx+1) * self.batch_size
        if r_bound > self.size():
            r_bound = self.size()
            l_bound = r_bound - self.batch_size
        batches = r_bound - l_bound

        # input images
        x_batch = np.zeros((batches, net_h, net_w, 3))
        # list of groundtruth boxes
        t_batch = np.zeros((batches, 1, 1, 1, self.max_box_per_image, 4))

        # initialize the inputs and the outputs
        grid_shapes = input_shape // self.downsample
        yolos = [
            self._get_grid_shape(1, grid_shapes),
            self._get_grid_shape(2, grid_shapes),
            self._get_grid_shape(4, grid_shapes)
        ]
        dummy1 = dummy2 = dummy3 = np.zeros((batches, 1))

        box_data = []
        # do the logic to fill in the inputs and the output
        for batch, image_data in enumerate(self.data[l_bound:r_bound]):
            # augment input image and fix object's position and size
            img, boxes = self._aug_image(image_data, net_shape)
            # assign input image to x_batch
            x_batch[batch] = self.norm(img)
            box_data.append(boxes)
        box_data = np.array(box_data)

        # true boxes
        true_boxes = np.array(box_data, dtype='float32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = (true_boxes[..., 2:4] - true_boxes[..., 0:2])
        true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

        # anchor
        anchors = np.array(self.raw_anchors).reshape(-1, 2)
        anchors = np.expand_dims(anchors, 0)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchor_max = anchors / 2.
        anchor_min = -anchor_max
        valid_mask = boxes_wh[..., 0] > 0

        for batch in range(self.batch_size):
            wh = boxes_wh[batch, valid_mask[batch]]
            if len(wh) == 0:
                continue
            wh = np.expand_dims(wh, -2)
            box_max = wh / 2.
            box_min = -box_max

            intersect_min = np.maximum(box_min, anchor_min)
            intersect_max = np.minimum(box_max, anchor_max)
            intersect_wh = np.maximum(intersect_max - intersect_min, 0)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for layer in range(self.num_layers):
                    if n not in anchor_mask[layer]:
                        continue

                    j = np.floor(
                        true_boxes[batch, t, 0] * yolos[layer].shape[2]
                    ).astype('int32')
                    i = np.floor(
                        true_boxes[batch, t, 1] * yolos[layer].shape[1]
                    ).astype('int32')
                    k = anchor_mask[layer].index(n)
                    c = true_boxes[batch, t, 4].astype('int32')

                    yolos[layer][batch, j, i, k, 0:4] = true_boxes[batch, t, 0:4]
                    yolos[layer][batch, j, i, k, 4] = 1
                    yolos[layer][batch, j, i, k, 5+c] = 1
                    t_batch[batch, 0, 0, 0, t] = [
                        true_boxes[batch, t, 0],
                        true_boxes[batch, t, 1],
                        box_data[batch][t][2] - box_data[batch][t][0],
                        box_data[batch][t][3] - box_data[batch][t][1],
                    ]

        return [x_batch, t_batch, *yolos], [dummy1, dummy2, dummy3]

    def _get_grid_shape(self, scale, grid):
        grid_h, grid_w = scale * grid
        grid_shape = np.zeros((
            self.batch_size,
            grid_h,
            grid_w,
            self.num_layers,
            self.num_classes
        ), dtype='float32')
        return grid_shape

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(
                self.min_net_size/self.downsample,
                self.max_net_size/self.downsample+1)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _aug_image(self, data, net_shape):
        image_path = data['path']
        image = cv2.imread(image_path)
        image = image[:, :, ::-1]  # BGR to RGB image

        image_h, image_w, _ = image.shape
        h, w = net_shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h
        new_ar = (image_w + np.random.uniform(-dw, dw)) / \
                 (image_h + np.random.uniform(-dh, dh))

        scale = np.random.uniform(0.25, 2)

        if (new_ar < 1):
            nh = int(scale * h)
            nw = int(h * new_ar)
        else:
            nw = int(scale * w)
            nh = int(w / new_ar)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

        dx = int(np.random.uniform(0, w - nw))
        dy = int(np.random.uniform(0, h - nh))
        # apply scaling and cropping
        image = apply_random_scale_and_crop(
           image, nw, nh, w, h, dx, dy)

        # randomly distort hsv space
        image = random_distort_image(image)

        # randomly flip
        flip = np.random.randint(2)
        image = random_flip(image, flip)

        # correct the size and pos of bounding boxes
        box = self._correct_bounding_boxes(
            data['object'], nw, nh, w, h,
            dx, dy, flip, image_w, image_h)

        return image, box

    def _correct_bounding_boxes(
        self, boxes, nw, nh, w, h, dx, dy, flip, iw, ih
    ):
        max_boxes = self.max_box_per_image
        box_data = np.zeros((max_boxes, 5))

        if len(boxes) > 0:
            # randomize boxes' order
            np.random.shuffle(boxes)

            # correct sizes and positions
            box = np.array([[
                int(b['xmin']),
                int(b['ymin']),
                int(b['xmax']),
                int(b['ymax']),
                self.labels.index(b['name'])
            ] for b in boxes])

            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            if flip == 1:
                box[:, [0, 2]] = w - box[:, [2, 0]]

            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # discard invalid box
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box_data[:len(box)] = box

        return box_data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.data)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.data[i]['object']:
            annot = [
                obj['xmin'],
                obj['ymin'],
                obj['xmax'],
                obj['ymax'],
                self.labels.index(obj['name'])
            ]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.data[i]['path'])
