import cv2
import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from .utils import get_random_data, preprocess_true_boxes
import matplotlib.pyplot as plt


def split(data, split_rate=0.9, shuffle_seed=10101):
    '''data split to train and valid'''
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)
    np.random.seed(None)
    split_index = int(len(data)*split_rate)

    train_data = data[:split_index]
    valid_data = data[split_index:]

    print(">> "
          f"Train on {len(train_data)} samples, "
          f"Valid on {len(valid_data)} samples ")

    return train_data, valid_data


class BatchGenerator(Sequence):
    def __init__(
        self, data, anchors, labels, downsample=32, max_box_per_image=30,
        batch_size=1, min_net_size=320, max_net_size=608,
        shuffle=True, jitter=True, norm=None
    ):
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.min_net_size = (min_net_size//self.downsample)*self.downsample
        self.max_net_size = (max_net_size//self.downsample)*self.downsample
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1])
                        for i in range(len(anchors)//2)]
        self.net_h = 416
        self.net_w = 416

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
        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))
        # list of groundtruth boxes
        t_batch = np.zeros((
            r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))

        # initialize the inputs and the outputs
        # desired network output 1
        yolo_1 = np.zeros((
            r_bound - l_bound, 1*base_grid_h,  1*base_grid_w,
            len(self.anchors)//3, 4+1+len(self.labels)))
        # desired network output 2
        yolo_2 = np.zeros((
            r_bound - l_bound, 2*base_grid_h,  2*base_grid_w,
            len(self.anchors)//3, 4+1+len(self.labels)))
        # desired network output 3
        yolo_3 = np.zeros((
            r_bound - l_bound, 4*base_grid_h,  4*base_grid_w,
            len(self.anchors)//3, 4+1+len(self.labels)))
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy1 = np.zeros((r_bound - l_bound, 1))
        dummy2 = np.zeros((r_bound - l_bound, 1))
        dummy3 = np.zeros((r_bound - l_bound, 1))

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.data[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(
                    0, 0, obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
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
                w = np.log((obj['xmax'] - obj['xmin']) /
                           float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) /
                           float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the loc of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x,
                     max_index % 3, 5+obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] -
                            obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

            # assign input image to x_batch
            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(
                        img,
                        (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                        (255, 0, 0), 3)
                    cv2.putText(
                        img, obj['name'], (obj['xmin']+2, obj['ymin']+12),
                        0, 1.2e-3 * img.shape[0], (0, 255, 0), 2)

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy1, dummy2, dummy3]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample,
                                                         self.max_net_size/self.downsample+1)
            # print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _aug_image(self, data, net_h, net_w):
        image_path, items = data.split(" ", maxsplit=1)
        image_boxes = [item.replace("\n", "") for item in items.split(" ")]
        image = cv2.imread(image_path)  # RGB image

        if image is None:
            print('Image Cannot find ', image_path)
            return

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
        try:
            im_sized = apply_random_scale_and_crop(
                image, new_w, new_h, net_w, net_h, dx, dy)
        except Exception:
            print()
            print("params : ", new_w, new_h, net_w, net_h, dx, dy)
            print(image_path)
            print(image_boxes)
            raise

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(
            image_boxes, new_w, new_h, net_w, net_h,
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
            annot = [obj['xmin'], obj['ymin'], obj['xmax'],
                     obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.data[i]['filename'])


def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):

    if new_w == 0:
        # print("SIZE ERR : new_w")
        new_w = 1
    if new_h == 0:
        # print("SIZE ERR : new_h")
        new_h = 1

    im_sized = cv2.resize(image, (new_w, new_h))

    if dx > 0:
        im_sized = np.pad(
            im_sized, ((0, 0), (dx, 0), (0, 0)),
            mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:, -dx:, :]

    if (new_w + dx) < net_w:
        im_sized = np.pad(
            im_sized, ((0, 0), (0, net_w - (new_w+dx)), (0, 0)),
            mode='constant', constant_values=127)

    if dy > 0:
        im_sized = np.pad(
            im_sized, ((dy, 0), (0, 0), (0, 0)),
            mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:, :, :]

    if (new_h + dy) < net_h:
        im_sized = np.pad(
            im_sized, ((0, net_h - (new_h+dy)), (0, 0), (0, 0)),
            mode='constant', constant_values=127)

    return im_sized[:net_h, :net_w, :]


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
        return cv2.flip(image, 1)
    return image


def correct_bounding_boxes(
    image_boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h
):
    # randomize boxes' order
    np.random.shuffle(image_boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    boxes = dict()
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax, name = boxes[i].split(",")
        boxes[i]['xmin'] = int(_constrain(0, net_w, xmin * sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, xmax * sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, ymin * sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, ymax * sy + dy))
        boxes[i]['name'] = name

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or \
           boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1:
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes


def _constrain(min_v, max_v, value):
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value
