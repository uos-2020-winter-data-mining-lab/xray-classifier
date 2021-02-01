import cv2
import numpy as np
from keras.utils import Sequence


class BatchGenerator(Sequence):
    def __init__(
        self,
        data,
        anchors,
        labels,
        downsample=32,
        max_box_per_image=30,
        batch_size=16,
        min_net_size=320,
        max_net_size=320,
        net_shape=(416, 416),
        shuffle=True,
        jitter=0.2,
    ):
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.num_classes = 4 + 1 + len(self.labels)

        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.min_net_size = min_net_size  # //self.downsample)*self.downsample
        self.max_net_size = max_net_size  # //self.downsample)*self.downsample
        self.jitter = jitter
        self.anchors = anchors
        self.num_layers = 3  # len(self.anchors) // 3
        self.net_h, self.net_w = net_shape

        if shuffle:
            np.random.shuffle(self.data)

    def __len__(self):
        return int(np.ceil(float(len(self.data))/self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        input_shape = np.array((net_h, net_w), dtype='int32')
        base_grid_h = net_h//self.downsample
        base_grid_w = net_w//self.downsample

        # determine the first and the last indices of the b
        r_bound = min((idx+1) * self.batch_size, len(self.data))
        l_bound = r_bound - self.batch_size

        # input images
        x_batch = np.zeros((self.batch_size, net_h, net_w, 3))
        # list of groundtruth boxes
        t_batch = np.zeros(
            (self.batch_size, 1, 1, 1, self.max_box_per_image, 4))

        # initialize the inputs and the outputs
        yolos = [
            self._get_grid_shape(1*base_grid_h, 1*base_grid_w),  # yolo1
            self._get_grid_shape(2*base_grid_h, 2*base_grid_w),  # yolo2
            self._get_grid_shape(4*base_grid_h, 4*base_grid_w)   # yolo3
        ]

        dummies = [
            np.zeros((self.batch_size, 1)),
            np.zeros((self.batch_size, 1)),
            np.zeros((self.batch_size, 1))
        ]

        box_data = []
        # do the logic to fill in the inputs and the output
        for b, batch_data in enumerate(self.data[l_bound:r_bound]):
            # augment input image and fix object's position and size
            x_batch[b], boxes = self._aug_image(batch_data, net_h, net_w)
            box_data.append(boxes)

        # true boxes
        true_boxes = np.array(box_data, dtype='float64')
        boxes_xy = (true_boxes[..., 2:4] + true_boxes[..., 0:2]) / 2
        boxes_wh = (true_boxes[..., 2:4] - true_boxes[..., 0:2])
        true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

        # anchor
        anchors = np.array(self.anchors).reshape(-1, 2)
        anchors = np.expand_dims(anchors, 0)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchor_max = anchors / 2.
        anchor_min = -anchor_max
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(self.batch_size):
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue
            wh = np.expand_dims(wh, -2)
            box_max = wh / 2.
            box_min = -box_max

            intersect_max = np.minimum(box_max, anchor_max)
            intersect_min = np.maximum(box_min, anchor_min)
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

                    grid_y = int(np.floor(
                        true_boxes[b, t, 1] * yolos[layer].shape[1]))
                    grid_x = int(np.floor(
                        true_boxes[b, t, 0] * yolos[layer].shape[2]))
                    anchor_box = anchor_mask[layer].index(n)
                    c = true_boxes[b, t, 4].astype('int32')

                    center_x = true_boxes[b, t, 0] * yolos[layer].shape[1]
                    center_y = true_boxes[b, t, 1] * yolos[layer].shape[2]

                    box_logw = np.log((wh[t, 0, 0]/float(anchors[..., n, 0])))
                    box_logh = np.log((wh[t, 0, 1]/float(anchors[..., n, 1])))
                    pred_box = [center_x, center_y, box_logw, box_logh]

                    true_w, true_h = box_data[b][t][2:4]-box_data[b][t][0:2]
                    true_box = [center_x, center_y, true_w, true_h]

                    yolos[layer][b, grid_y, grid_x, anchor_box, 0:4] = pred_box
                    yolos[layer][b, grid_y, grid_x, anchor_box, 4] = 1
                    yolos[layer][b, grid_y, grid_x, anchor_box, 5+c] = 1
                    t_batch[b, 0, 0, 0, t] = true_box

        return [x_batch, t_batch, *yolos], [*dummies]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(
                self.min_net_size/self.downsample,
                self.max_net_size/self.downsample+1)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _get_grid_shape(self, grid_h, grid_w):
        grid_shape = np.zeros((
            self.batch_size, grid_h, grid_w,
            self.num_layers, self.num_classes
        ))
        return grid_shape

    def _aug_image(self, data, net_h, net_w):
        image = cv2.imread(data['path'])

        image = image[:, :, ::-1]  # RGB image
        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        scale = np.random.uniform(0.25, 2)
        new_ar = (image_w + np.random.uniform(-dw, dw)) / \
            (image_h + np.random.uniform(-dh, dh))

        if (new_ar < 1):
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)
        image = cv2.resize(image, (new_w, new_h))

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        image = apply_random_scale_and_crop(
            image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        # image = random_distort_image(image)

        # randomly flip
        flip = np.random.randint(2)
        image = random_flip(image, flip)
        image = image / 255.

        # correct the size and pos of bounding boxes
        data = self.correct_bounding_boxes(
            data['object'], new_w, new_h, net_w, net_h,
            dx, dy, flip, image_w, image_h)

        return image, data

    def correct_bounding_boxes(
        self, boxes, nw, nh, w, h, dx, dy, flip, iw, ih
    ):
        max_boxes = self.max_box_per_image
        box_data = np.zeros((max_boxes, 5))

        # randomize boxes' order
        np.random.shuffle(boxes)
        # correct sizes and positions

        box = np.array([[
            b['xmin'], b['ymin'], b['xmax'], b['ymax'],
            self.labels.index(b['name'])
        ] for b in boxes]).astype('int32')

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

    def size(self):
        return len(self.data)

    def on_epoch_end(self):
        np.random.shuffle(self.data)
    """


    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    """
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

        return np.array(annots).astype('int32')

    def load_image(self, i):
        return cv2.imread(self.data[i]['path'])


def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    if dx > 0:
        image = np.pad(
            image, ((0, 0), (dx, 0), (0, 0)), mode='edge')
    else:
        image = image[:, -dx:, :]

    if (new_w + dx) < net_w:
        image = np.pad(
            image, ((0, 0), (0, net_w - (new_w+dx)), (0, 0)), mode='edge')

    if dy > 0:
        image = np.pad(
            image, ((dy, 0), (0, 0), (0, 0)), mode='edge')
    else:
        image = image[-dy:, :, :]

    if (new_h + dy) < net_h:
        image = np.pad(
            image, ((0, net_h - (new_h+dy)), (0, 0), (0, 0)), mode='edge')

    return image[:net_h, :net_w, :]


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180)*180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1./scale


def random_flip(image, flip):
    if flip == 1:
        return np.flip(image, (0, 1))
    return image
