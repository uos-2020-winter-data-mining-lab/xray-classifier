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
        self.num_anchorbox = len(self.anchors) // 3
        self.net_h, self.net_w = net_shape

        if shuffle:
            np.random.shuffle(self.data)

    def __len__(self):
        return int(np.ceil(float(len(self.data))/self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h = net_h // self.downsample
        base_grid_w = net_w // self.downsample

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
        yolo1 = self._get_grid_shape(1, base_grid_h, base_grid_w)
        yolo2 = self._get_grid_shape(2, base_grid_h, base_grid_w)
        yolo3 = self._get_grid_shape(4, base_grid_h, base_grid_w)
        yolos = [yolo3, yolo2, yolo1]
        dummy1 = dummy2 = dummy3 = np.zeros((batches, 1))

        data_count = 0
        true_box_index = 0
        # do the logic to fill in the inputs and the output
        for train_data in self.data[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_data, net_h, net_w)

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                obj_w, obj_h = obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin']
                shifted_box = BoundBox(0, 0, obj_w, obj_h)

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if iou > max_iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index//3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5*(obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5*(obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log(obj_w / float(max_anchor.xmax))  # t_w
                h = np.log(obj_h / float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the loc of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                cur_index = max_index % 3
                yolo[data_count, grid_y, grid_x, cur_index % 3] = 0
                yolo[data_count, grid_y, grid_x, cur_index % 3, 0:4] = box
                yolo[data_count, grid_y, grid_x, cur_index % 3, 4] = 1.
                yolo[data_count, grid_y, grid_x, cur_index % 3, 5+obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj_w, obj_h]
                t_batch[data_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

            # assign input image to x_batch
            x_batch[data_count] = self.norm(img)

            # increase data counter in the current batch
            data_count += 1

        return [x_batch, t_batch, yolo1, yolo2, yolo3], [dummy1, dummy2, dummy3]

    def _get_grid_shape(self, scale, grid_h, grid_w):
        grid_shape = np.zeros((
            self.batch_size,
            scale * grid_h,
            scale * grid_w,
            self.num_anchorbox,
            self.num_classes
        ))
        return grid_shape

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(
                self.min_net_size/self.downsample,
                self.max_net_size/self.downsample+1)
            # print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

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
        # im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(
            data['object'], new_w, new_h, net_w, net_h,
            dx, dy, flip, image_w, image_h)

        return im_sized, all_objs

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
