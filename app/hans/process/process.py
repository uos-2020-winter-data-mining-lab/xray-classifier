import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from .utils import get_random_data, preprocess_true_boxes
from app.hans.config import WIDTH, HEIGHT, RATIO, INPUT_SHAPE


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


def preprocess_data(data):
    print(f"total : {len(data)}")
    for i, row in enumerate(data):
        try:
            source_path = row.split(" ")[0]
            cropped_img = crop_and_resize_images(source_path)

            data[i] = row.replace("raw\\", "img\\")
            target_path = source_path.replace("raw\\", "img\\")
            paths = target_path.split("\\")[:-1]
            cur_path = ''
            for path in paths:
                cur_path = os.path.join(cur_path, path)
                try:
                    os.mkdir(cur_path)
                except FileExistsError:
                    pass

            cropped_img.save(target_path)
        except Exception:
            print(data[0])
            raise
    return data


def crop_and_resize_images(source_path, resize_size=INPUT_SHAPE):
    rgb_image = cv2.imread(source_path, cv2.IMREAD_COLOR)
    rgb_image = rgb_image[50:HEIGHT - 100, 100:WIDTH-100]
    gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    _, thresh = cv2.threshold(
        gray_img, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    idx = 0
    ls_xmin = []
    ls_ymin = []
    ls_xmax = []
    ls_ymax = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        ls_xmin.append(x)
        ls_ymin.append(y)
        ls_xmax.append(x + w)
        ls_ymax.append(y + h)
    xmin = min(ls_xmin)
    ymin = min(ls_ymin)
    xmax = max(ls_xmax)
    ymax = max(ls_ymax)

    roi = rgb_image[ymin:ymax, xmin:xmax]

    resized_roi = cv2.resize(roi, INPUT_SHAPE)
    resized_img = resized_roi.reshape((320, 320, 3))

    cropped_img = Image.fromarray(resized_img)

    return cropped_img


def generator(data, input_shape, batch_size, anchors, num_classes):
    '''data generator for fit_generator'''
    N = len(data)
    if N == 0 or batch_size <= 0:
        return None

    i = 0
    while True:
        image_data = list()
        box_data = list()

        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data)
            image, box = get_random_data(data[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % N

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(
            box_data, input_shape, anchors, num_classes)

        yield [image_data, *y_true], np.zeros(batch_size)
