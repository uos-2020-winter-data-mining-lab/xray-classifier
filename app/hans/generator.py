import cv2
import numpy as np
from keras.utils import Sequence
from app.hans.bbox import BoundBox, bbox_iou
from .generator_util import (
    apply_random_scale_and_crop,
    random_distort_image,
    random_flip,
    correct_bounding_boxes
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
        net_h, net_w = self._get_net_size(idx)
        base_grid_h = net_h//self.downsample
        base_grid_w = net_w//self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.data):
            r_bound = len(self.data)
            l_bound = r_bound - self.batch_size

        # input images
        x_batch = np.zeros((self.batch_size, net_h, net_w, 3))
        # list of groundtruth boxes
        t_batch = np.zeros(
            (self.batch_size, 1, 1, 1, self.max_box_per_image, 4))

        # initialize the inputs and the outputs
        yolo1 = np.zeros((self.batch_size, 1*base_grid_h,  1*base_grid_w,
                          self.num_layers, self.num_classes))
        yolo2 = np.zeros((self.batch_size, 2*base_grid_h,  2*base_grid_w,
                          self.num_layers, self.num_classes))
        yolo3 = np.zeros((self.batch_size, 4*base_grid_h,  4*base_grid_w,
                          self.num_layers, self.num_classes))
        yolos = [yolo1, yolo2, yolo3]

        dummies = [
            np.zeros((self.batch_size, 1)),
            np.zeros((self.batch_size, 1)),
            np.zeros((self.batch_size, 1))
        ]

        # do the logic to fill in the inputs and the output
        for batch, batch_data in enumerate(self.data[l_bound:r_bound]):
            # augment input image and fix object's position and size
            img_data, box_data = self._aug_image(batch_data, net_h, net_w)

            # assign input image to x_batch
            x_batch[batch] = self.norm(img_data)

            for t, obj in enumerate(box_data):
                if (obj[2] - obj[0]) * (obj[3]-obj[1]) == 0:
                    continue
                xmin, ymin, xmax, ymax, label_idx = obj

                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0, 0, xmax-xmin, ymax-ymin)

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                # determine the yolo to be responsible for this bounding box
                yolo = yolos[2 - max_index//3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5*(xmin + xmax)
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5*(ymin + ymax)
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((xmax - xmin)/float(max_anchor.xmax))
                h = np.log((ymax - ymin)/float(max_anchor.ymax))

                box = [center_x, center_y, w, h]

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[batch, grid_y, grid_x, max_index % 3] = 0
                yolo[batch, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[batch, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[batch, grid_y, grid_x, max_index % 3, 5+int(label_idx)] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, xmax - xmin, ymax - ymin]
                t_batch[batch, 0, 0, 0, t] = true_box

        yolos = [yolo1, yolo2, yolo3]
        return [x_batch, t_batch, *yolos], [*dummies]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(
                self.min_net_size/self.downsample,
                self.max_net_size/self.downsample+1)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _get_grid_shape(self, scale, grid_h, grid_w):
        grid_shape = np.zeros((
            self.batch_size,
            scale * grid_h,
            scale * grid_w,
            self.num_anchorbox,
            self.num_classes
        ))
        return grid_shape

    def _aug_image(self, data, net_h, net_w):
        image_path = data['path']
        image = cv2.imread(image_path)

        image = image[:, :, ::-1]  # RGB image
        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / \
            (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)

        if (new_ar < 1):
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(
            image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        data = self._correct_bounding_boxes(
            data['object'], new_w, new_h, net_w, net_h,
            dx, dy, flip, image_w, image_h)

        return im_sized, data

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
                int(self.labels.index(b['name']))
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
